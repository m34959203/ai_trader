from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd

# Внешние зависимости:
#   pip install ccxt pandas
import ccxt

# Убедись, что PYTHONPATH включает .../ai_trader/analysis/
# Например, перед запуском:
#   set PYTHONPATH=C:\Users\Дмитрий\Desktop\ai_trader
# или запусти из корня проекта: python -m scripts.demo_analyze ...
from analysis.analyze_market import analyze_market, DEFAULT_CONFIG, AnalysisConfig  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ohlcv_to_df(data: List[List[Any]]) -> pd.DataFrame:
    """
    CCXT OHLCV → DataFrame с индексом (UTC):
      columns: open, high, low, close, volume
    """
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(data, columns=cols)
    # TZ-aware UTC индекс
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    # Приводим к float
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def fetch_ohlcv_safe(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
    max_retries: int = 5,
    pause: float = 1.0,
) -> List[List[Any]]:
    """Безопасная загрузка OHLCV с ретраями и легким экспоненциальным бэкоффом."""
    attempt = 0
    while True:
        try:
            return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except (ccxt.NetworkError, ccxt.DDoSProtection, ccxt.ExchangeNotAvailable) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = pause * (2 ** (attempt - 1))
            print(f"[WARN] {e.__class__.__name__}: {e}. Retry {attempt}/{max_retries} in {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)
        except ccxt.BaseError as e:
            # Другие ошибки (например, неверный символ/таймфрейм)
            raise


def pretty_print_result(res: dict) -> None:
    # Короткий свод + JSON
    trend = res.get("trend")
    vol = res.get("volatility")
    sig = res.get("signal")
    conf = res.get("confidence")
    mtf = res.get("mtf", {})
    print("\n=== SUMMARY ===")
    print(f"Trend:       {trend}")
    print(f"Volatility:  {vol}")
    print(f"Signal:      {sig}")
    print(f"Confidence:  {conf}")
    if mtf:
        print(f"MTF(4h):     {mtf.get('trend_4h')}")
    # Причины
    reasons = res.get("reasons", [])
    if reasons:
        print("\nReasons:")
        for i, r in enumerate(reasons, 1):
            print(f"  {i}. {r}")
    # Топ уровни
    levels = res.get("levels", [])
    if levels:
        print("\nTop Levels:")
        for lv in levels[:8]:
            print(f"  {lv['kind']:11s} @ {lv['price']:.4f}  strength={lv['strength']}")
    # Полный JSON
    print("\n=== RAW JSON ===")
    print(json.dumps(res, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-Trader · Demo Market Analysis via CCXT")
    p.add_argument("--exchange", default="binance", help="Биржа CCXT (по умолчанию: binance)")
    p.add_argument("--symbol", default="BTC/USDT", help="Торговый символ CCXT (например, BTC/USDT)")
    p.add_argument("--tf-fast", default="1h", help="Таймфрейм для быстрого анализа (по умолчанию: 1h)")
    p.add_argument("--tf-slow", default="4h", help="Таймфрейм для подтверждения (по умолчанию: 4h)")
    p.add_argument("--limit-fast", type=int, default=360, help="Кол-во баров fast TF (>=120)")
    p.add_argument("--limit-slow", type=int, default=240, help="Кол-во баров slow TF (>=50)")
    p.add_argument("--no-mtf", action="store_true", help="Отключить MTF подтверждение")
    p.add_argument("--save-csv", action="store_true", help="Сохранить загруженные данные в CSV")
    p.add_argument("--save-json", action="store_true", help="Сохранить результат анализа в JSON")
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "out"),
                   help="Папка для сохранения файлов")
    # Переопределения некоторых параметров анализа (опционально)
    p.add_argument("--buy-th", type=int, default=None, help="Порог buy (по умолчанию 60)")
    p.add_argument("--sell-th", type=int, default=None, help="Порог sell (по умолчанию 40)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    cfg = DEFAULT_CONFIG
    # Перекрываем только заданные пользователем пороги
    if args.buy_th is not None or args.sell_th is not None:
        cfg = AnalysisConfig(
            **{
                **cfg.__dict__,
                **({"buy_threshold": args.buy_th} if args.buy_th is not None else {}),
                **({"sell_threshold": args.sell_th} if args.sell_th is not None else {}),
            }
        )
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    # Инициализируем биржу
    ex_cls = getattr(ccxt, args.exchange, None)
    if ex_cls is None:
        print(f"[ERROR] Unknown exchange: {args.exchange}", file=sys.stderr)
        sys.exit(2)
    ex: ccxt.Exchange = ex_cls({
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })

    # Загрузка OHLCV
    try:
        data_fast = fetch_ohlcv_safe(ex, args.symbol, args.tf_fast, args.limit_fast)
        df_fast = ohlcv_to_df(data_fast)
        df_slow = None
        if not args.no_mtf:
            data_slow = fetch_ohlcv_safe(ex, args.symbol, args.tf_slow, args.limit_slow)
            df_slow = ohlcv_to_df(data_slow)
    except Exception as e:
        print(f"[ERROR] Failed to fetch OHLCV: {e}", file=sys.stderr)
        sys.exit(1)

    # Сохранение исходных данных при желании
    outdir = Path(args.outdir).resolve()
    if args.save_csv:
        outdir.mkdir(parents=True, exist_ok=True)
        fast_csv = outdir / f"{args.exchange}_{args.symbol.replace('/', '-')}_{args.tf_fast}.csv"
        df_fast.to_csv(fast_csv)
        print(f"[INFO] Saved: {fast_csv}")
        if df_slow is not None:
            slow_csv = outdir / f"{args.exchange}_{args.symbol.replace('/', '-')}_{args.tf_slow}.csv"
            df_slow.to_csv(slow_csv)
            print(f"[INFO] Saved: {slow_csv}")

    # Запускаем анализ
    try:
        res = analyze_market(df_fast, df_slow, config=cfg)
    except AssertionError as e:
        print(f"[ERROR] Analysis assertion: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        sys.exit(4)

    # Результат
    pretty_print_result(res)

    # Сохранить JSON при желании
    if args.save_json:
        outdir.mkdir(parents=True, exist_ok=True)
        out_json = outdir / f"analysis_{args.exchange}_{args.symbol.replace('/', '-')}_{args.tf_fast}.json"
        out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved: {out_json}")


if __name__ == "__main__":
    main()
