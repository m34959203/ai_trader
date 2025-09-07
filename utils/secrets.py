# utils/secrets.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, List

import yaml  # обязательная зависимость по проекту

# python-dotenv — опционально. Если установлен, подхватим .env автоматически.
try:  # pragma: no cover
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


# =============================================================================
# Helpers
# =============================================================================

def _exists(p: Optional[str]) -> bool:
    return bool(p) and os.path.exists(p)  # type: ignore[arg-type]


def _read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not _exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


def _parse_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _redact(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 6:
        return "***"
    return s[:3] + "…" + s[-3:]


def _env_get_any(names: List[str]) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Глубокое слияние словарей (override поверх base) для совместимости с exec.test.yaml.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out


def _ensure_map(obj: Any) -> dict:
    """
    Мягкая нормализация раздела конфигурации: если раздел не dict (bool/str/num),
    возвращаем пустую map, чтобы не падать на .items().
    """
    if obj is None or isinstance(obj, dict):
        return obj or {}
    return {}


def _to_ms(value: Any, default_ms: int) -> int:
    """
    Конвертация seconds→ms (если значение < 1000 считаем секундами),
    иначе воспринимаем как миллисекунды.
    """
    try:
        v = float(value)
        return int(v * 1000) if v < 1000 else int(v)
    except Exception:
        return default_ms


# =============================================================================
# dotenv auto-loading
# =============================================================================

def load_env_files(*, prefer_test: Optional[bool] = None) -> None:
    """
    Пытается загрузить переменные окружения из .env-файлов.
    Ничего не делает, если python-dotenv не установлен.

    Приоритет поиска (первый найденный файл загружается, остальные игнорируются):
    1) путь из ENV_FILE или UVICORN_ENV_FILE
    2) .env.test (если prefer_test=True) или .env
    3) ./configs/.env.test, ./configs/.env
    """
    if load_dotenv is None:  # библиотека не установлена — пропускаем
        return

    explicit = os.getenv("ENV_FILE") or os.getenv("UVICORN_ENV_FILE")
    if _exists(explicit):
        load_dotenv(explicit, override=False)
        return

    prefer_test = prefer_test if prefer_test is not None else _parse_bool(os.getenv("TESTNET"), default=True)
    if prefer_test:
        candidates = [".env.test", ".env", os.path.join("configs", ".env.test"), os.path.join("configs", ".env")]
    else:
        candidates = [".env", ".env.test", os.path.join("configs", ".env"), os.path.join("configs", ".env.test")]

    for p in candidates:
        if _exists(p):
            load_dotenv(p, override=False)
            break


# =============================================================================
# exec.yaml — единый конфиг
# =============================================================================

@lru_cache(maxsize=1)
def load_exec_config() -> Dict[str, Any]:
    """
    Грузим единый конфиг исполнения (предпочтительно один файл exec.yaml).

    Приоритет:
      1) EXEC_CONFIG=path/to/exec.yaml (если задан)
      2) exec.yaml (в корне проекта)
      3) configs/exec.yaml

    Обратная совместимость:
      - Если обнаружен exec.test.yaml (или configs/exec.test.yaml), он аккуратно
        смержится поверх основной конфигурации (НО это больше не требуется).
    """
    main_candidates = [
        os.getenv("EXEC_CONFIG"),
        "exec.yaml",
        os.path.join("configs", "exec.yaml"),
    ]
    cfg_main: Dict[str, Any] = {}
    for p in main_candidates:
        cfg_main = _read_yaml(p)
        if cfg_main:
            break

    # устаревший, но безопасный overlay
    legacy_candidates = [
        "exec.test.yaml",
        os.path.join("configs", "exec.test.yaml"),
    ]
    cfg_legacy: Dict[str, Any] = {}
    for p in legacy_candidates:
        data = _read_yaml(p)
        if data:
            cfg_legacy = _merge_dicts(cfg_legacy, data)

    cfg = _merge_dicts(cfg_main, cfg_legacy) if cfg_legacy else cfg_main

    # ── Мягкая нормализация разделов (защита от 'bool'.items()) ──
    if not isinstance(cfg, dict):
        # Если верхний уровень не map — вернём пустой dict (и пусть API сообщит человеку)
        return {}

    for section in ("binance", "risk", "sim", "limits", "strategy", "ui"):
        if section in cfg and not isinstance(cfg[section], dict):
            cfg[section] = _ensure_map(cfg[section])
            cfg.setdefault("_config_warnings", []).append(
                f"exec.yaml: раздел '{section}' был не-объектом; автоматически нормализован в {{}}"
            )

    return cfg


def _binance_node(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Возвращает секцию binance{} из exec.yaml (или пустую).
    Допустимые схемы:
      binance:
        api_key: ...
        api_secret: ...
        base_url: ...
        recv_window: ...
        timeout: ...
        max_retries: ...
        backoff_base: ...
        backoff_cap: ...
        testnet: { api_key, api_secret, base_url, ... }
        main|prod: { api_key, api_secret, base_url, ... }
        profiles: { testnet: {...}, main|prod: {...} }
    """
    return (cfg.get("binance") or {}) if isinstance(cfg, dict) else {}


def _profile_view(node: Dict[str, Any], *, testnet: bool) -> Dict[str, Any]:
    """
    Возвращает профильную ветку из binance{} с учётом поддерживаемых схем.
    Профильные поля перекрывают корневые.
    """
    prof_key = "testnet" if testnet else "main"
    profiles = node.get("profiles") if isinstance(node, dict) else None
    prof = {}
    if isinstance(profiles, dict):
        prof = profiles.get(prof_key) or profiles.get("prod") or {}

    if not prof:
        if testnet:
            prof = node.get("testnet") or {}
        else:
            prof = node.get("main") or node.get("prod") or {}

    # Корневые дефолты
    root = dict(node)
    for k in ("profiles", "testnet", "main", "prod"):
        if isinstance(root.get(k), dict):
            root.pop(k)

    # root <- prof
    return _merge_dicts(root, prof or {})


# =============================================================================
# Binance: ключи и конфиг
# =============================================================================

def get_binance_config(*, testnet: bool) -> Dict[str, Any]:
    """
    Возвращает объединённый конфиг профиля Binance из exec.yaml (без секретов из ENV).
    """
    cfg = load_exec_config()
    node = _binance_node(cfg)
    return _profile_view(node, testnet=testnet)


def get_binance_base_url(*, testnet: bool) -> str:
    """
    Выбор base_url: приоритет exec.yaml → ENV → дефолт.
    """
    prof = get_binance_config(testnet=testnet)
    url = prof.get("base_url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    env_url = os.getenv("BINANCE_BASE")
    if env_url and env_url.strip():
        return env_url.strip()
    return "https://testnet.binance.vision" if testnet else "https://api.binance.com"


def get_binance_keys(*, testnet: bool) -> Tuple[str, str]:
    """
    Возвращает (api_key, api_secret) для Binance.

    Приоритет источников:
      1) ENV (включая алиасы)
      2) exec.yaml (единый файл, разные допустимые структуры, см. _profile_view)
      3) -> RuntimeError с понятным сообщением
    """
    # Подгружаем .env автоматически (если есть dotenv)
    load_env_files(prefer_test=testnet)

    # 1) ENV с поддержкой алиасов
    if testnet:
        api_key = _env_get_any(["BINANCE_TESTNET_API_KEY", "BINANCE_API_KEY_TESTNET", "BINANCE_KEY_TESTNET", "BINANCE_API_KEY"])
        api_sec = _env_get_any(["BINANCE_TESTNET_API_SECRET", "BINANCE_API_SECRET_TESTNET", "BINANCE_SECRET_TESTNET", "BINANCE_API_SECRET"])
    else:
        api_key = _env_get_any(["BINANCE_API_KEY", "BINANCE_KEY"])
        api_sec = _env_get_any(["BINANCE_API_SECRET", "BINANCE_SECRET"])

    # 2) Единый exec.yaml (если в ENV не нашли)
    if not api_key or not api_sec:
        prof = get_binance_config(testnet=testnet)
        api_key = api_key or prof.get("api_key")
        api_sec = api_sec or prof.get("api_secret")

    if not api_key or not api_sec:
        env_hint = "BINANCE_TESTNET_API_KEY/SECRET или BINANCE_API_KEY/SECRET" if testnet else "BINANCE_API_KEY/SECRET"
        cfg_hint = "binance.[profiles.testnet|testnet].api_key(.api_secret) или binance.api_key(.api_secret)"
        raise RuntimeError(
            "Binance API keys not found. "
            f"Provide {env_hint} in environment (or .env/.env.test) "
            f"or set {cfg_hint} in exec.yaml."
        )

    return str(api_key), str(api_sec)


def get_ccxt_kwargs(*, testnet: bool) -> Dict[str, Any]:
    """
    Готовые kwargs для ccxt.binance с корректными URL для spot testnet
    и подхватом настроек из exec.yaml (recv_window, timeout и т.п.).
    """
    api_key, api_secret = get_binance_keys(testnet=testnet)
    prof = get_binance_config(testnet=testnet)

    recv_window_ms = _to_ms(prof.get("recv_window", 20000), 20000)
    timeout_ms = _to_ms(prof.get("timeout", 20), 20000)  # 20 сек по умолчанию
    base_url = get_binance_base_url(testnet=testnet)

    kwargs: Dict[str, Any] = {
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "timeout": timeout_ms,
        "options": {
            "defaultType": "spot",
            "adjustForTimeDifference": True,
            "recvWindow": recv_window_ms,
        },
    }

    if testnet:
        # ccxt для spot сам не переключает URL → принудительно задаём testnet
        kwargs["urls"] = {
            "api": {
                "public": f"{base_url}/api",
                "private": f"{base_url}/api",
            }
        }
    return kwargs


def get_retry_policy() -> Dict[str, Any]:
    """
    Возвращает политику повторов для сетевых ошибок (используется в вызовах executors.api_binance).
    """
    cfg = load_exec_config()
    b = cfg.get("binance", {}) if isinstance(cfg, dict) else {}
    return {
        "max_retries": int(b.get("max_retries", 5)),
        "backoff_base": float(b.get("backoff_base", 0.5)),
        "backoff_cap": float(b.get("backoff_cap", 8.0)),
    }


# =============================================================================
# Public profiles / diagnostics (без секретов)
# =============================================================================

def current_exec_profile(*, mode: str, testnet: bool) -> Dict[str, Any]:
    """
    Возвращает информацию для эндпоинта /exec/config (без секретов).
    Показывает, откуда будут браться ключи, базовые runtime-параметры и найден ли exec.yaml.
    """
    load_env_files(prefer_test=testnet)
    cfg = load_exec_config()
    prof = get_binance_config(testnet=testnet) if mode == "binance" else {}

    # Источник ключей
    key_ok, sec_ok = False, False
    key_src, sec_src = None, None
    red_key, red_sec = "", ""

    if testnet:
        k_env = _env_get_any(["BINANCE_TESTNET_API_KEY", "BINANCE_API_KEY_TESTNET", "BINANCE_KEY_TESTNET", "BINANCE_API_KEY"])
        s_env = _env_get_any(["BINANCE_TESTNET_API_SECRET", "BINANCE_API_SECRET_TESTNET", "BINANCE_SECRET_TESTNET", "BINANCE_API_SECRET"])
    else:
        k_env = _env_get_any(["BINANCE_API_KEY", "BINANCE_KEY"])
        s_env = _env_get_any(["BINANCE_API_SECRET", "BINANCE_SECRET"])

    if k_env:
        key_ok, key_src, red_key = True, "env", _redact(k_env)
    if s_env:
        sec_ok, sec_src, red_sec = True, "env", _redact(s_env)

    # Если не нашли в ENV — проверим exec.yaml
    if not (key_ok and sec_ok):
        k_cfg = prof.get("api_key")
        s_cfg = prof.get("api_secret")
        if k_cfg and not key_ok:
            key_ok, key_src, red_key = True, "exec.yaml", _redact(k_cfg)
        if s_cfg and not sec_ok:
            sec_ok, sec_src, red_sec = True, "exec.yaml", _redact(s_cfg)

    runtime_cfg = {
        "base_url": prof.get("base_url") or get_binance_base_url(testnet=testnet),
        "recv_window": prof.get("recv_window"),
        "timeout": prof.get("timeout"),
        "max_retries": prof.get("max_retries"),
        "backoff_base": prof.get("backoff_base"),
        "backoff_cap": prof.get("backoff_cap"),
        "ticker_ttl": prof.get("ticker_ttl"),
        "exchange_info_ttl": prof.get("exchange_info_ttl"),
    }

    return {
        "mode": mode,
        "testnet": testnet,
        "binance_keys": {
            "present": bool(key_ok and sec_ok),
            "api_key": red_key,
            "api_secret": "***" if sec_ok else "",
            "api_key_source": key_src or "missing",
            "api_secret_source": sec_src or "missing",
        },
        "config_loaded": bool(cfg),
        "warnings": cfg.get("_config_warnings", []),
        "runtime": runtime_cfg,
    }


def ensure_binance_env_loaded(*, testnet: bool) -> None:
    """
    Удобный вызов перед созданием BinanceExecutor:
    пытается загрузить .env и проверяет наличие ключей (бросит исключение, если их нет).
    """
    load_env_files(prefer_test=testnet)
    # это гарантированно бросит понятную ошибку, если ключей нет
    get_binance_keys(testnet=testnet)
