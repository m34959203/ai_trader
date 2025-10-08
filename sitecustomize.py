"""
Compatibility shim for httpx>=0.28: support AsyncClient(app=...) used by legacy tests.

- httpx 0.28 убрал параметр `app=` у Client/AsyncClient:
  теперь надо передавать transport=ASGITransport(app=app).
  Док: https://www.python-httpx.org/advanced/transports/
- Модуль `site` автоматически пытается импортировать `sitecustomize` при старте
  интерпретатора (если не запущено с -S). Это делает шим глобальным.
"""

from __future__ import annotations

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

if httpx is not None and not getattr(httpx, "_ai_trader_httpx_app_shim", False):
    try:
        _orig_async_init = httpx.AsyncClient.__init__  # type: ignore[attr-defined]

        def _shim_asyncclient_init(self, *args, **kwargs):
            # Перехватываем устаревший аргумент `app=...` и превращаем его в transport=ASGITransport(app=...)
            app = kwargs.pop("app", None)
            if app is not None and "transport" not in kwargs:
                try:
                    transport = httpx.ASGITransport(app=app)  # type: ignore[attr-defined]
                    kwargs["transport"] = transport
                    # Для старых тестов полезен base_url по умолчанию
                    kwargs.setdefault("base_url", "http://testserver")
                except Exception:
                    # В крайнем случае дадим оригинальной инициализации упасть — тест покажет причину
                    pass
            return _orig_async_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _shim_asyncclient_init  # type: ignore[assignment]
        httpx._ai_trader_httpx_app_shim = True  # type: ignore[attr-defined]
    except Exception:
        # Ничего: если нет AsyncClient (не должен быть так), не вмешиваемся
        pass
