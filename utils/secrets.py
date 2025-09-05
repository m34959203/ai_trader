# utils/secrets.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

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


def _env_get_any(names: list[str]) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _cfg_get(d: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
    """
    d['binance']['testnet']['api_key']  => _cfg_get(d, 'binance.testnet.api_key')
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


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

    return _merge_dicts(cfg_main, cfg_legacy) if cfg_legacy else cfg_main


def _binance_node(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Возвращает секцию binance{} из exec.yaml (или пустую).
    Поддерживает варианты:
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
        profiles: { testnet: {...}, main: {...} }
    """
    return (cfg.get("binance") or {}) if isinstance(cfg, dict) else {}


def _profile_view(node: Dict[str, Any], *, testnet: bool) -> Dict[str, Any]:
    """
    Возвращает профильную ветку из binance{} с учётом поддерживаемых схем.
    Профильные поля перекрывают корневые.
    """
    # 1) внутри 'profiles'
    prof_key = "testnet" if testnet else "main"
    profiles = node.get("profiles") if isinstance(node, dict) else None
    prof = {}
    if isinstance(profiles, dict):
        prof = profiles.get(prof_key) or profiles.get("prod") or {}

    # 2) плоская схема: binance.testnet / binance.main|prod
    if not prof:
        if testnet:
            prof = node.get("testnet") or {}
        else:
            prof = node.get("main") or node.get("prod") or {}

    # 3) root уровень — дефолты (api_key, api_secret, base_url, etc.)
    root = dict(node)

    # Безопасно удалить вложенные карты, чтобы они не попали «как есть»
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
    Полезно для инициализации executors.api_binance.BinanceExecutor:
    возвращает объединённый конфиг для текущего профиля из единого exec.yaml
    (без секретов из ENV).

    Возвращаемые поля (если заданы в YAML):
      - base_url, recv_window, timeout, max_retries, backoff_base, backoff_cap, ticker_ttl, exchange_info_ttl, ...
      - api_key, api_secret (если храните их в YAML — не рекомендуется для прод)
    """
    cfg = load_exec_config()
    node = _binance_node(cfg)
    return _profile_view(node, testnet=testnet)


def get_binance_base_url(*, testnet: bool) -> str:
    """
    Выбор base_url: приоритет exec.yaml → дефолт Binance.
    """
    prof = get_binance_config(testnet=testnet)
    url = prof.get("base_url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    return "https://testnet.binance.vision" if testnet else "https://api.binance.com"


def get_binance_keys(*, testnet: bool) -> Tuple[str, str]:
    """
    Возвращает (api_key, api_secret) для Binance.

    Приоритет источников:
      1) ENV (включая алиасы)
      2) exec.yaml (единый файл, разные допустимые структуры, см. _profile_view)
      3) -> RuntimeError с понятным сообщением
    """
    # 0) Подгружаем .env автоматически (если есть dotenv)
    load_env_files(prefer_test=testnet)

    # 1) ENV с поддержкой алиасов
    if testnet:
        api_key = _env_get_any(["BINANCE_TESTNET_API_KEY", "BINANCE_API_KEY_TESTNET", "BINANCE_KEY_TESTNET"])
        api_sec = _env_get_any(["BINANCE_TESTNET_API_SECRET", "BINANCE_API_SECRET_TESTNET", "BINANCE_SECRET_TESTNET"])
    else:
        api_key = _env_get_any(["BINANCE_API_KEY", "BINANCE_KEY"])
        api_sec = _env_get_any(["BINANCE_API_SECRET", "BINANCE_SECRET"])

    # 2) Единый exec.yaml (если в ENV не нашли)
    if not api_key or not api_sec:
        prof = get_binance_config(testnet=testnet)
        # Приоритет: узел профиля → корень
        api_key = api_key or prof.get("api_key")
        api_sec = api_sec or prof.get("api_secret")

    if not api_key or not api_sec:
        env_hint = "BINANCE_TESTNET_API_KEY/SECRET" if testnet else "BINANCE_API_KEY/SECRET"
        cfg_hint = "binance.[profiles.testnet|testnet].api_key(.api_secret) или binance.api_key(.api_secret)"
        raise RuntimeError(
            "Binance API keys not found. "
            f"Provide {env_hint} in environment (or .env/.env.test) "
            f"or set {cfg_hint} in exec.yaml."
        )

    return str(api_key), str(api_sec)


# =============================================================================
# Public profiles / diagnostics (без секретов)
# =============================================================================

def current_exec_profile(*, mode: str, testnet: bool) -> Dict[str, Any]:
    """
    Возвращает информацию для эндпоинта /exec/config (без секретов).
    Показывает, откуда будут браться ключи, базовые runtime-параметры и найден ли exec.yaml.
    """
    # Подтягиваем env (помогает для корректного статуса)
    load_env_files(prefer_test=testnet)
    cfg = load_exec_config()
    prof = get_binance_config(testnet=testnet) if mode == "binance" else {}

    # Источник ключей
    key_ok, sec_ok = False, False
    key_src, sec_src = None, None
    red_key, red_sec = "", ""

    if testnet:
        k_env = _env_get_any(["BINANCE_TESTNET_API_KEY", "BINANCE_API_KEY_TESTNET", "BINANCE_KEY_TESTNET"])
        s_env = _env_get_any(["BINANCE_TESTNET_API_SECRET", "BINANCE_API_SECRET_TESTNET", "BINANCE_SECRET_TESTNET"])
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

    # Небезопасные поля из профиля (без секретов), полезны для диагностики
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
