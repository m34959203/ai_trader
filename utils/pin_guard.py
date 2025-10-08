import os, time
import pyotp

PIN_ENV = "STARTUP_PIN"
TOTP_ENV = "STARTUP_TOTP_SECRET"

def verify_pin(pin: str) -> bool:
    expected = os.getenv(PIN_ENV, "")
    return bool(expected) and (pin == expected)

def verify_totp(code: str) -> bool:
    secret = os.getenv(TOTP_ENV, "")
    if not secret:
        return False
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)
