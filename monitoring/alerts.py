from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

LOG = logging.getLogger("ai_trader.alerts")

# ──────────────────────────────────────────────────────────────────────────────
# Конфиг: ENV → configs/exec.{yaml,json} (ENV имеет приоритет)
# ──────────────────────────────────────────────────────────────────────────────

def _load_from_file(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        if path.suffix.lower() == ".json":
            return json.loads(text) or {}
    except Exception as e:
        LOG.debug("alerts: failed to read %s: %r", path, e)
    return {}

def _find_config_file() -> Optional[Path]:
    # ожидаем структуру ai_trader/configs/exec.yaml
    base = Path(__file__).resolve().parents[1] / "configs"
    for name in ("exec.yaml", "exec.yml", "exec.json"):
        p = base / name
        if p.exists():
            return p
    return None

def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _as_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _as_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

@dataclass(frozen=True)
class TelegramCfg:
    enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    parse_mode: str = "Markdown"
    disable_preview: bool = True
    timeout_sec: int = 15

@dataclass(frozen=True)
class SmtpCfg:
    enabled: bool = False
    host: Optional[str] = None
    port: int = 465
    user: Optional[str] = None
    password: Optional[str] = None
    from_addr: Optional[str] = None
    to_addrs: Tuple[str, ...] = ()
    use_ssl: bool = True           # SMTPS (465)
    starttls: bool = False         # STARTTLS (обычно 587)
    timeout_sec: int = 20

@dataclass(frozen=True)
class AlertsConfig:
    service: str = os.getenv("APP_NAME", "ai-trader")
    version: str = os.getenv("APP_VERSION", "unknown")
    env: str = os.getenv("APP_ENV", os.getenv("ENV", "dev"))
    hostname: str = socket.gethostname()
    telegram: TelegramCfg = TelegramCfg()
    smtp: SmtpCfg = SmtpCfg()
    # троттлинг/повторные попытки
    min_interval_sec: float = float(os.getenv("ALERTS_MIN_INTERVAL_SEC", "0.5"))
    max_retries: int = _as_int_env("ALERTS_MAX_RETRIES", 3)
    backoff_base: float = float(os.getenv("ALERTS_BACKOFF_BASE", "0.8"))

def load_alerts_config() -> AlertsConfig:
    file_cfg = _load_from_file(_find_config_file())

    # TELEGRAM
    tg = TelegramCfg(
        enabled=_as_bool_env("TELEGRAM_ENABLED", _get_nested(file_cfg, "alerts", "telegram", "enabled", default=True)),
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN", _get_nested(file_cfg, "alerts", "telegram", "bot_token")),
        chat_id=os.getenv("TELEGRAM_CHAT_ID", _get_nested(file_cfg, "alerts", "telegram", "chat_id")),
        parse_mode=os.getenv("TELEGRAM_PARSE_MODE", _get_nested(file_cfg, "alerts", "telegram", "parse_mode", default="Markdown")),
        disable_preview=_as_bool_env("TELEGRAM_DISABLE_PREVIEW", _get_nested(file_cfg, "alerts", "telegram", "disable_preview", default=True)),
        timeout_sec=_as_int_env("TELEGRAM_TIMEOUT_SEC", _get_nested(file_cfg, "alerts", "telegram", "timeout_sec", default=15)),
    )

    # SMTP
    to_raw = os.getenv("SMTP_TO", None)
    if to_raw:
        to_list = tuple(x.strip() for x in to_raw.split(",") if x.strip())
    else:
        _file_to = _get_nested(file_cfg, "alerts", "smtp", "to", default=[]) or []
        to_list = tuple(str(x).strip() for x in _file_to if str(x).strip())

    smtp = SmtpCfg(
        enabled=_as_bool_env("SMTP_ENABLED", _get_nested(file_cfg, "alerts", "smtp", "enabled", default=False)),
        host=os.getenv("SMTP_HOST", _get_nested(file_cfg, "alerts", "smtp", "host")),
        port=_as_int_env("SMTP_PORT", int(_get_nested(file_cfg, "alerts", "smtp", "port", default=465))),
        user=os.getenv("SMTP_USER", _get_nested(file_cfg, "alerts", "smtp", "user")),
        password=os.getenv("SMTP_PASSWORD", _get_nested(file_cfg, "alerts", "smtp", "password")),
        from_addr=os.getenv("SMTP_FROM", _get_nested(file_cfg, "alerts", "smtp", "from")),
        to_addrs=to_list,
        use_ssl=_as_bool_env("SMTP_USE_SSL", _get_nested(file_cfg, "alerts", "smtp", "use_ssl", default=True)),
        starttls=_as_bool_env("SMTP_STARTTLS", _get_nested(file_cfg, "alerts", "smtp", "starttls", default=False)),
        timeout_sec=_as_int_env("SMTP_TIMEOUT_SEC", _get_nested(file_cfg, "alerts", "smtp", "timeout_sec", default=20)),
    )

    cfg = AlertsConfig(
        service=os.getenv("APP_NAME", _get_nested(file_cfg, "app", "name", default="ai-trader")),
        version=os.getenv("APP_VERSION", _get_nested(file_cfg, "app", "version", default="unknown")),
        env=os.getenv("APP_ENV", os.getenv("ENV", _get_nested(file_cfg, "app", "env", default="dev"))),
        hostname=socket.gethostname(),
        telegram=tg,
        smtp=smtp,
        min_interval_sec=float(os.getenv("ALERTS_MIN_INTERVAL_SEC", _get_nested(file_cfg, "alerts", "min_interval_sec", default=0.5))),
        max_retries=_as_int_env("ALERTS_MAX_RETRIES", _get_nested(file_cfg, "alerts", "max_retries", default=3)),
        backoff_base=float(os.getenv("ALERTS_BACKOFF_BASE", _get_nested(file_cfg, "alerts", "backoff_base", default=0.8))),
    )
    return cfg

CFG = load_alerts_config()

# ──────────────────────────────────────────────────────────────────────────────
# Состояние отправки: троттлинг и клиенты
# ──────────────────────────────────────────────────────────────────────────────

_last_send_ts: float = 0.0
_throttle_lock = asyncio.Lock()

# Бестолезное падение отправки нам не нужно — будем ретраить
async def _async_retry(fn, *, retries: int, backoff_base: float, name: str):
    last_exc = None
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            delay = backoff_base * (2 ** attempt)
            LOG.warning("alerts: %s send failed, retry %d/%d: %r", name, attempt + 1, retries, e)
            await asyncio.sleep(delay)
    if last_exc:
        raise last_exc

async def _throttle():
    global _last_send_ts
    async with _throttle_lock:
        now = time.time()
        dt = now - _last_send_ts
        if dt < CFG.min_interval_sec:
            await asyncio.sleep(CFG.min_interval_sec - dt)
        _last_send_ts = time.time()

# ──────────────────────────────────────────────────────────────────────────────
# Telegram sender (aiohttp/httpx/requests fallback)
# ──────────────────────────────────────────────────────────────────────────────

async def _send_telegram(text: str) -> None:
    if not (CFG.telegram.enabled and CFG.telegram.bot_token and CFG.telegram.chat_id):
        return

    # Telegram ограничение ~4096 символов
    if len(text) > 4000:
        text = text[:3990] + "…"

    url = f"https://api.telegram.org/bot{CFG.telegram.bot_token}/sendMessage"
    payload = {
        "chat_id": CFG.telegram.chat_id,
        "text": text,
        "parse_mode": CFG.telegram.parse_mode,
        "disable_web_page_preview": CFG.telegram.disable_preview,
    }

    async def _do_aiohttp():
        import aiohttp  # type: ignore
        timeout = aiohttp.ClientTimeout(total=CFG.telegram.timeout_sec)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json=payload) as resp:
                if resp.status // 100 != 2:
                    body = await resp.text()
                    raise RuntimeError(f"telegram http {resp.status}: {body[:200]}")

    async def _do_httpx():
        import httpx  # type: ignore
        async with httpx.AsyncClient(timeout=CFG.telegram.timeout_sec) as client:
            r = await client.post(url, json=payload)
            if r.status_code // 100 != 2:
                raise RuntimeError(f"telegram http {r.status_code}: {r.text[:200]}")

    async def _do_requests():
        import requests  # type: ignore
        def _sync():
            r = requests.post(url, json=payload, timeout=CFG.telegram.timeout_sec)
            if r.status_code // 100 != 2:
                raise RuntimeError(f"telegram http {r.status_code}: {r.text[:200]}")
        await asyncio.to_thread(_sync)

    # выбираем доступный транспорт
    try:
        import aiohttp  # noqa: F401
        sender = _do_aiohttp
    except Exception:
        try:
            import httpx  # noqa: F401
            sender = _do_httpx
        except Exception:
            sender = _do_requests

    await _async_retry(sender, retries=CFG.max_retries, backoff_base=CFG.backoff_base, name="telegram")

# ──────────────────────────────────────────────────────────────────────────────
# SMTP sender (smtplib в threadpool; SSL/STARTTLS)
# ──────────────────────────────────────────────────────────────────────────────

def _smtp_send_sync(subject: str, html: str, plain: str) -> None:
    if not CFG.smtp.enabled:
        return
    if not (CFG.smtp.host and CFG.smtp.from_addr and CFG.smtp.to_addrs):
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = CFG.smtp.from_addr
    msg["To"] = ", ".join(CFG.smtp.to_addrs)

    part1 = MIMEText(plain, "plain", "utf-8")
    part2 = MIMEText(html, "html", "utf-8")
    msg.attach(part1)
    msg.attach(part2)

    import smtplib  # stdlib
    if CFG.smtp.use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(CFG.smtp.host, CFG.smtp.port, timeout=CFG.smtp.timeout_sec, context=context) as s:
            if CFG.smtp.user and CFG.smtp.password:
                s.login(CFG.smtp.user, CFG.smtp.password)
            s.sendmail(CFG.smtp.from_addr, list(CFG.smtp.to_addrs), msg.as_string())
    else:
        with smtplib.SMTP(CFG.smtp.host, CFG.smtp.port, timeout=CFG.smtp.timeout_sec) as s:
            if CFG.smtp.starttls:
                s.starttls(context=ssl.create_default_context())
            if CFG.smtp.user and CFG.smtp.password:
                s.login(CFG.smtp.user, CFG.smtp.password)
            s.sendmail(CFG.smtp.from_addr, list(CFG.smtp.to_addrs), msg.as_string())

async def _send_email(subject: str, html: str, plain: str) -> None:
    if not CFG.smtp.enabled:
        return
    await _async_retry(
        lambda: asyncio.to_thread(_smtp_send_sync, subject, html, plain),
        retries=CFG.max_retries,
        backoff_base=CFG.backoff_base,
        name="smtp",
    )

# ──────────────────────────────────────────────────────────────────────────────
# Публичный API
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_prefix(level: str) -> Tuple[str, str]:
    # emoji, subject tag
    level = level.lower()
    if level == "crit":
        return "🛑 CRITICAL", "[CRIT]"
    if level == "warn":
        return "⚠️ WARNING", "[WARN]"
    return "ℹ️ INFO", "[INFO]"

def _mk_text(level: str, msg: str, context: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    """
    Возвращает (telegram_text, email_subject, email_plain/html)
    """
    prefix, subj_tag = _fmt_prefix(level)
    ctx = context or {}
    meta = {
        "service": CFG.service,
        "version": CFG.version,
        "env": CFG.env,
        "host": CFG.hostname,
        **ctx,
    }

    # Telegram — Markdown by default
    t_lines = [
        f"*{prefix}*",
        f"*{CFG.service}* v{CFG.version} · `{CFG.env}` · `{CFG.hostname}`",
        "",
        msg,
    ]
    if ctx:
        try:
            pretty = "```\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n```"
            t_lines += ["", pretty]
        except Exception:
            pass
    telegram_text = "\n".join(t_lines)

    subject = f"{subj_tag} {CFG.service}@{CFG.env}: {msg[:120]}"

    # Email plain + html
    plain = f"{prefix}\n{CFG.service} v{CFG.version} · {CFG.env} · {CFG.hostname}\n\n{msg}\n"
    if ctx:
        try:
            plain += "\n" + json.dumps(ctx, ensure_ascii=False, indent=2) + "\n"
        except Exception:
            pass

    html_ctx = ""
    if ctx:
        try:
            html_ctx = f"<pre>{json.dumps(ctx, ensure_ascii=False, indent=2)}</pre>"
        except Exception:
            pass

    html = f"""
    <html><body>
      <h3>{prefix}</h3>
      <div><b>{CFG.service}</b> v{CFG.version} · <code>{CFG.env}</code> · <code>{CFG.hostname}</code></div>
      <p style="white-space:pre-wrap">{msg}</p>
      {html_ctx}
    </body></html>
    """.strip()

    return telegram_text, subject, (plain, html)

async def _alert(level: str, msg: str, *, context: Optional[Dict[str, Any]] = None) -> None:
    # троттлинг, чтобы не DOS-ить каналы
    await _throttle()
    tg_text, subj, (plain, html) = _mk_text(level, msg, context=context)

    # Параллельно отправляем во все включенные каналы
    tasks = []
    if CFG.telegram.enabled:
        tasks.append(_send_telegram(tg_text))
    if CFG.smtp.enabled:
        tasks.append(_send_email(subj, html, plain))

    if not tasks:
        # Нет сконфигурированных каналов — логируем
        LOG.warning("alerts: no channels configured; message=%s", msg)
        return

    try:
        await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        LOG.error("alerts: send failed: %r", e)

# ——— Удобные ярлыки ————————————————————————————————————————————————
async def alert_info(msg: str, *, context: Optional[Dict[str, Any]] = None) -> None:
    await _alert("info", msg, context=context)

async def alert_warn(msg: str, *, context: Optional[Dict[str, Any]] = None) -> None:
    await _alert("warn", msg, context=context)

async def alert_crit(msg: str, *, context: Optional[Dict[str, Any]] = None) -> None:
    await _alert("crit", msg, context=context)

async def alert_exception(exc: BaseException, *, where: str = "", context: Optional[Dict[str, Any]] = None) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    msg = f"Exception {type(exc).__name__} at {where or 'unknown'}\n{tb}"
    ctx = dict(context or {})
    await alert_crit(msg, context=ctx)

# ——— Доп. уведомления для жизненного цикла ——————————————
async def alert_startup() -> None:
    await alert_info("Service startup", context={"cfg": asdict(CFG)})

async def alert_shutdown() -> None:
    await alert_warn("Service shutdown")

# ──────────────────────────────────────────────────────────────────────────────
# Быстрый self-test (ручной): python -m ai_trader.monitoring.alerts
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _main():
        LOG.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        LOG.addHandler(h)
        await alert_info("Test info", context={"k": 1})
        await alert_warn("Test warn", context={"k": 2})
        await alert_crit("Test crit", context={"k": 3})
        try:
            1 / 0
        except Exception as e:
            await alert_exception(e, where="__main__")
    asyncio.run(_main())
