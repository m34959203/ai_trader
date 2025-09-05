from __future__ import annotations
from typing import List, Dict, Optional, Literal

Side = Literal["long", "short"]


class PaperTrader:
    """
    Модель «одна позиция одновременно» с поддержкой:
      • long/short
      • SL/TP + приоритет SL над TP (gap-through учитывается по high/low на баре)
      • трейлинг-стопа (по close)
      • учёта комиссии при входе/выходе
      • отчёта (winrate, pnl_sum, avg_pnl, max_dd по закрытым сделкам)
      • детального журнала сделок в формате, совместимом со схемами API:
            entry_ts, exit_ts, entry_price, exit_price, qty, side,
            pnl, ret_pct, fees, sl_hit, tp_hit, bars_held, notes

    Правила сигналов:
      - signal=+1: открыть/удерживать позицию в направлении self.side (если нет открытой)
      - signal=-1: закрыть позицию (переворот трактуем как закрытие)
      - signal=0 : игнор
    Приоритет выходов на баре: SL → TP (учитываем gap-through).
    Трейлинг активируется на закрытии бара: сначала проверяем срабатывания
    по текущим стопам, затем подтягиваем SL по close (эффект — со следующего бара).
    """

    # ────────────────────────────────────────────────────────────────────────
    # Инициализация
    # ────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
        fee_pct: float = 0.001,
        trail_pct: Optional[float] = None,  # 0.01 = 1% трейлинг
        side: Side = "long",                # направление сигналов +1
        qty: float = 1.0,
        price_epsilon: float = 1e-9,        # числовая устойчивость сравнений
    ):
        self.sl_pct = float(sl_pct)
        self.tp_pct = float(tp_pct)
        self.fee_pct = float(fee_pct)
        # важная деталь: 0.0 означает «выключено», >0 — включено
        self.trail_pct = None if trail_pct is None else float(trail_pct)
        self.side: Side = side
        self.qty = max(0.0, float(qty))
        self.eps = float(price_epsilon)

        # Журнал закрытых сделок (list[dict])
        self.trades: List[Dict] = []

        # Текущая позиция (или None). Для long:
        #   {"entry_ts","entry_price","max_price","sl","tp","fee_in","trail_active","bars_held"}
        # Для short:
        #   {"entry_ts","entry_price","min_price","sl","tp","fee_in","trail_active","bars_held"}
        self.position: Optional[Dict] = None

        # аккуратная статистика комиссий
        self._fees_in_total: float = 0.0
        self._fees_out_total: float = 0.0

    # ────────────────────────────────────────────────────────────────────────
    # Вход/выход по сигналу
    # ────────────────────────────────────────────────────────────────────────
    def on_signal(self, ts: int, price: float, signal: int) -> None:
        """
        signal:
          +1 — открыть/держать позицию в направлении self.side (если пусто)
          -1 — закрыть позицию
           0 — игнор
        """
        price = self._sanitize_price(price)
        if price is None or self.qty <= 0.0:
            return

        # вход
        if self.position is None and signal == 1:
            self._open(ts, price)
            return

        # закрытие
        if self.position is not None and signal == -1:
            self.close(ts, price, reason="signal")

    # ────────────────────────────────────────────────────────────────────────
    # Универсальный шаг: бар + сигнал (удобно для бэктестов)
    # ────────────────────────────────────────────────────────────────────────
    def step_bar(
        self,
        ts: int,
        o: float,
        h: float,
        l: float,
        c: float,
        signal: int = 0,
        check_first: Literal["stops", "signal"] = "stops",
    ) -> None:
        """
        Обработка одного бара. По умолчанию сначала проверяем стопы (stops), затем сигнал.
        Это безопаснее (если на баре словили SL — считаем, что вышли до нового входа).
        """
        if check_first == "stops":
            self.check_sl_tp(ts, h, l, close=c)
            if self.position is None:
                if signal == 1:
                    self.on_signal(ts, c, signal)
            else:
                if signal == -1:
                    self.on_signal(ts, c, signal)
        else:
            self.on_signal(ts, c, signal)
            self.check_sl_tp(ts, h, l, close=c)

        # учёт удержания позиции по времени
        if self.position is not None:
            self.position["bars_held"] = int(self.position.get("bars_held", 0)) + 1

    # ────────────────────────────────────────────────────────────────────────
    # Принудительное закрытие
    # ────────────────────────────────────────────────────────────────────────
    def close(self, ts: int, price: float, reason: str = "signal") -> None:
        if not self.position:
            return

        price = self._sanitize_price(price)
        if price is None:
            return

        entry = float(self.position["entry_price"])
        fee_in = float(self.position.get("fee_in", 0.0))
        fee_out = float(price) * self.fee_pct * self.qty

        if self.side == "long":
            gross = (float(price) - entry) * self.qty
        else:
            gross = (entry - float(price)) * self.qty  # прибыль по шорту при падении

        pnl = gross - fee_in - fee_out
        ret_pct = pnl / (entry * self.qty) if entry > 0 and self.qty > 0 else 0.0

        sl_hit = "SL" in reason.upper()
        tp_hit = reason.upper() == "TP"

        trade_rec = {
            "entry_ts": int(self.position["entry_ts"]),
            "entry_price": entry,
            "exit_ts": int(ts),
            "exit_price": float(price),
            "pnl": float(pnl),
            "ret_pct": float(ret_pct),
            "fees": float(fee_in + fee_out),
            "fee_in": float(fee_in),
            "fee_out": float(fee_out),
            "reason": reason,
            "trail": bool(self.position.get("trail_active", False)),
            "sl_hit": bool(sl_hit),
            "tp_hit": bool(tp_hit),
            "side": self.side,
            "qty": float(self.qty),
            "bars_held": int(self.position.get("bars_held", 0)),
            "notes": None,
        }
        self.trades.append(trade_rec)

        self._fees_out_total += float(fee_out)
        self.position = None

    # ────────────────────────────────────────────────────────────────────────
    # Проверка SL/TP/трейлинга на баре
    # ────────────────────────────────────────────────────────────────────────
    def check_sl_tp(self, ts: int, high: float, low: float, close: Optional[float] = None) -> None:
        """
        Проверка условий выхода на одном баре.

        Аргументы:
          ts   : метка времени бара (int)
          high : максимум бара
          low  : минимум бара
          close: цена закрытия бара (если не передать — возьмём mid=(high+low)/2)

        Порядок:
          1) проверяем срабатывания по текущим SL/TP (приоритет SL → TP; учитываем gap-through);
          2) если позиция ещё открыта — подтягиваем трейлинговый SL по close (если включён).
             Эффект трейлинга распространяется на последующие бары (не закрываем сразу на этом же).
        """
        if not self.position:
            return

        h = self._sanitize_price(high)
        l = self._sanitize_price(low)
        c = self._sanitize_price(close if close is not None else (high + low) / 2.0)
        if h is None or l is None or c is None:
            return

        # ---- 1) СРАБАТЫВАНИЯ ПО ТЕКУЩИМ СТОПАМ (до изменения трейла) ----
        cur_sl = self.position.get("sl")
        cur_tp = self.position.get("tp")

        if self.side == "long":
            # приоритет SL
            if cur_sl is not None and l <= float(cur_sl) + self.eps:
                self.close(ts, float(cur_sl), reason=("TRAIL_SL" if self.position.get("trail_active") else "SL"))
                return
            if cur_tp is not None and h >= float(cur_tp) - self.eps:
                self.close(ts, float(cur_tp), reason="TP")
                return
        else:  # short
            if cur_sl is not None and h >= float(cur_sl) - self.eps:
                self.close(ts, float(cur_sl), reason=("TRAIL_SL" if self.position.get("trail_active") else "SL"))
                return
            if cur_tp is not None and l <= float(cur_tp) + self.eps:
                self.close(ts, float(cur_tp), reason="TP")
                return

        # ---- 2) ПОДТЯГИВАЕМ ТРЕЙЛИНГ ПО CLOSE (если включён) ------------
        if self.trail_pct and self.trail_pct > 0.0:
            if self.side == "long":
                # монотонно обновляем максимум и SL
                prev_max = float(self.position.get("max_price", self.position["entry_price"]))
                new_max = max(prev_max, float(c))
                self.position["max_price"] = new_max
                trail_sl = new_max * (1.0 - float(self.trail_pct))
                # поднимаем SL только вверх
                old_sl = self.position.get("sl")
                self.position["sl"] = max(float(old_sl) if old_sl is not None else float("-inf"), trail_sl)
                self.position["trail_active"] = True
            else:
                # монотонно обновляем минимум и SL
                prev_min = float(self.position.get("min_price", self.position["entry_price"]))
                new_min = min(prev_min, float(c))
                self.position["min_price"] = new_min
                trail_sl = new_min * (1.0 + float(self.trail_pct))
                # опускаем SL только вниз (для short SL сверху, значит «вниз» — к меньшему числу)
                old_sl = self.position.get("sl")
                self.position["sl"] = min(float(old_sl) if old_sl is not None else float("+inf"), trail_sl)
                self.position["trail_active"] = True

        # ВАЖНО: не проверяем SL/TP повторно на этом же баре после подтягивания трейла.

    # ────────────────────────────────────────────────────────────────────────
    # Отчёт по сделкам
    # ────────────────────────────────────────────────────────────────────────
    def summary(self) -> Dict:
        """
        Возвращает агрегаты по закрытым сделкам:
            count   : кол-во сделок
            winrate : доля прибыльных
            pnl_sum : суммарный PnL (денежный)
            avg_pnl : средний PnL на сделку (денежный)
            max_dd  : максимальная просадка по суммарному PnL (денежная величина)
            fees_in : суммарные комиссии входа
            fees_out: суммарные комиссии выхода
        Примечание: max_dd здесь считается по кумулятивному PnL закрытых сделок
        (в деньгах). Относительная просадка по капиталу считается на уровне API.
        """
        if not self.trades:
            return {
                "count": 0,
                "winrate": 0.0,
                "pnl_sum": 0.0,
                "avg_pnl": 0.0,
                "max_dd": 0.0,
                "fees_in": 0.0,
                "fees_out": 0.0,
            }

        pnls = [float(t["pnl"]) for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)
        pnl_sum = float(sum(pnls))
        avg_pnl = pnl_sum / len(pnls)

        # простая оценка макс. просадки по закрытым сделкам (в деньгах)
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        return {
            "count": len(pnls),
            "winrate": wins / len(pnls),
            "pnl_sum": pnl_sum,
            "avg_pnl": avg_pnl,
            "max_dd": max_dd,
            "fees_in": float(self._fees_in_total),
            "fees_out": float(self._fees_out_total),
        }

    # ────────────────────────────────────────────────────────────────────────
    # Утилиты / сервисные методы
    # ────────────────────────────────────────────────────────────────────────
    def current_position(self) -> Optional[Dict]:
        """Копия текущей позиции (или None)."""
        return None if self.position is None else dict(self.position)

    def reset(self) -> None:
        """Полный сброс состояния."""
        self.trades.clear()
        self.position = None
        self._fees_in_total = 0.0
        self._fees_out_total = 0.0

    def update_params(
        self,
        *,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        trail_pct: Optional[float] = None,
        fee_pct: Optional[float] = None,
        qty: Optional[float] = None,
        side: Optional[Side] = None,
    ) -> None:
        """Мягкое обновление параметров (для долгих бэктестов/экспериментов)."""
        if sl_pct is not None:
            self.sl_pct = float(sl_pct)
        if tp_pct is not None:
            self.tp_pct = float(tp_pct)
        if trail_pct is not None:
            self.trail_pct = None if trail_pct is None else float(trail_pct)
        if fee_pct is not None:
            self.fee_pct = float(fee_pct)
        if qty is not None:
            self.qty = max(0.0, float(qty))
        if side is not None:
            self.side = side

    # ────────────────────────────────────────────────────────────────────────
    # Внутренние хелперы
    # ────────────────────────────────────────────────────────────────────────
    def _open(self, ts: int, price: float) -> None:
        """Открыть позицию согласно self.side."""
        if self.qty <= 0.0:
            return
        fee_in = float(price) * self.fee_pct * self.qty
        self._fees_in_total += float(fee_in)

        if self.side == "long":
            self.position = {
                "entry_ts": int(ts),
                "entry_price": float(price),
                "max_price": float(price),  # важно для трейлинга
                "sl": float(price * (1 - self.sl_pct)) if self.sl_pct else None,
                "tp": float(price * (1 + self.tp_pct)) if self.tp_pct else None,
                "fee_in": fee_in,
                "trail_active": False,
                "bars_held": 0,
            }
        else:
            self.position = {
                "entry_ts": int(ts),
                "entry_price": float(price),
                "min_price": float(price),  # важно для трейлинга
                "sl": float(price * (1 + self.sl_pct)) if self.sl_pct else None,  # стоп выше входа
                "tp": float(price * (1 - self.tp_pct)) if self.tp_pct else None,  # тейк ниже входа
                "fee_in": fee_in,
                "trail_active": False,
                "bars_held": 0,
            }

    def _sanitize_price(self, x: Optional[float]) -> Optional[float]:
        """Безопасно приводим цену к float, фильтруем NaN/inf/неположительные."""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if not (v == v) or v in (float("inf"), float("-inf")):
            return None
        if v <= 0.0:
            return None
        return v
