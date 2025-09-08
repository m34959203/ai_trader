# scripts/faucet_request.py
# -*- coding: utf-8 -*-
"""
Пополнение USDT на Binance Spot Testnet через /sapi/v1/faucet/capital.

Зависимости: requests (pip install requests)
Ключи берём из окружения (или .env, если используете python-dotenv).

ENV:
  BINANCE_TESTNET_API_KEY / BINANCE_API_KEY
  BINANCE_TESTNET_API_SECRET / BINANCE_API_SECRET

Запуск:
  python scripts/faucet_request.py --asset USDT --amount 10000
  # или просто:
  python scripts/faucet_request.py

После пополнения можно проверить баланс:
  python scripts/faucet_request.py --check
"""

import os
import hmac
import time
import json
import hashlib
import argparse
from urllib.parse import urlencode

try:
    import requests  # type: ignore
except Exception as e:
    raise SystemExit("Не установлен requests. Установите: pip install requests") from e


TESTNET_BASE = "https://testnet.binance.vision"


def _load_env_from_dotenv_if_present():
    # Опционально подхватить .env без зависимости, если есть
    path = ".env"
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass


def get_keys():
    _load_env_from_dotenv_if_present()
    api_key = (
        os.getenv("BINANCE_TESTNET_API_KEY")
        or os.getenv("BINANCE_API_KEY")
        or ""
    )
    api_secret = (
        os.getenv("BINANCE_TESTNET_API_SECRET")
        or os.getenv("BINANCE_API_SECRET")
        or ""
    )
    if not api_key or not api_secret:
        raise SystemExit(
            "Не заданы ключи Testnet.\n"
            "Установите переменные окружения BINANCE_TESTNET_API_KEY и BINANCE_TESTNET_API_SECRET\n"
            "или BINANCE_API_KEY / BINANCE_API_SECRET."
        )
    return api_key, api_secret


def sign(query_str: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), query_str.encode("utf-8"), hashlib.sha256).hexdigest()


def faucet_request(asset: str = "USDT", amount: int = 10_000) -> dict:
    api_key, api_secret = get_keys()
    ts = int(time.time() * 1000)

    params = {
        "asset": asset.upper(),
        "amount": amount,
        "timestamp": ts,
    }
    q = urlencode(params)
    sig = sign(q, api_secret)

    url = f"{TESTNET_BASE}/sapi/v1/faucet/capital?{q}&signature={sig}"
    headers = {"X-MBX-APIKEY": api_key}

    resp = requests.post(url, headers=headers, timeout=20)
    if resp.status_code >= 400:
        # Binance часто возвращает JSON с полями code/msg
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        raise SystemExit(
            f"Ошибка faucet {resp.status_code}: {json.dumps(data, ensure_ascii=False)}"
        )
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}


def get_account() -> dict:
    api_key, api_secret = get_keys()
    ts = int(time.time() * 1000)
    params = {"timestamp": ts}
    q = urlencode(params)
    sig = sign(q, api_secret)

    url = f"{TESTNET_BASE}/api/v3/account?{q}&signature={sig}"
    headers = {"X-MBX-APIKEY": api_key}

    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        raise SystemExit(
            f"Ошибка account {resp.status_code}: {json.dumps(data, ensure_ascii=False)}"
        )
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Binance Spot Testnet faucet helper")
    parser.add_argument("--asset", default="USDT", help="Актив для пополнения (по умолчанию USDT)")
    parser.add_argument("--amount", type=int, default=10000, help="Сумма пополнения (по умолчанию 10000)")
    parser.add_argument("--check", action="store_true", help="Только проверить /api/v3/account и вывести баланс")
    args = parser.parse_args()

    if args.check:
        acc = get_account()
        print(json.dumps(acc, indent=2, ensure_ascii=False))
        return

    data = faucet_request(asset=args.asset, amount=args.amount)
    print("Faucet ответ:")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    # Показать итоговый баланс после пополнения
    try:
        time.sleep(1.0)
        acc = get_account()
        print("\nТекущий account:")
        print(json.dumps(acc, indent=2, ensure_ascii=False))
    except SystemExit as e:
        print("\nПредупреждение при чтении баланса:", e)


if __name__ == "__main__":
    main()
