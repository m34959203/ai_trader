# Stage 2 (Advanced Market Intelligence) Readiness

## Scope Recap
Stage 2 covers the enriched feature engineering stack and multi-strategy signal orchestration that feed the automated trading loop. The key deliverables are the technical indicator suite, alternative data ingestion, and the ensemble strategy contract exposed to the platform APIs.

## What is Implemented
- **Broad indicator coverage.** `src/indicators.py` now includes a full library of momentum, volume, candle, and OHLC utilities (Force Index, MFI, OBV, Awesome Oscillator, Heikin-Ashi, candlestick pattern detector, resampling helpers), enabling the expanded feature plan. 【F:src/indicators.py†L1-L200】
- **Sentiment-aware market analysis.** `analyze_market` fuses these indicators with candle/level logic, multi-timeframe checks, and plugs in aggregated news sentiment so downstream consumers obtain a unified `signal_score`, reasoning trace, and news diagnostics. 【F:src/analysis/analyze_market.py†L1-L462】【F:src/analysis/analyze_market.py†L463-L760】
- **News processing pipeline.** `news/rss_client.py` fetches and scores RSS/atom feeds, while `news/nlp_gate.py` provides cached LLM-powered (with heuristic fallback) sentiment/importance extraction. The new `src/analysis/news_sentiment.py` aggregates headlines, weights them by impact, and returns normalized sentiment for the analyzer. 【F:news/rss_client.py†L1-L160】【F:news/nlp_gate.py†L1-L200】【F:src/analysis/news_sentiment.py†L1-L120】
- **Configurable strategy ensembles.** `/strategy/signals` can execute YAML-defined ensembles with frequency filters and return mode metadata, satisfying the contract for multi-strategy orchestration. 【F:routers/trading.py†L400-L520】【F:src/strategy.py†L1-L260】

## Outstanding Items
- **Real-time news ingestion wiring.** The analyzer expects fresh `data/news.json` (produced by `news/rss_client.py`), but there is no scheduled job/worker in this repo that keeps it current. Need to document or implement the ingestion trigger in deployment scripts so Stage 2 truly benefits from live alternative data.
- **Indicator regression coverage.** Existing tests cover a subset of the new indicators. Add targeted unit tests (Force Index, OBV, candlestick flags, resampling helpers) to guard against regressions when parameters evolve.
- **Strategy config governance.** Ensembles are loaded from YAML on demand, yet there is no validation schema or config-versioning story. Introduce JSONSchema/Pydantic validation and deployment guidance to prevent malformed configs from reaching production.

## Readiness Assessment
Functional deliverables are in place and integrated end-to-end, but the lack of an automated news ingestion loop and limited regression tests leave operational risk. **Stage 2 is ~90% complete**; addressing the outstanding bullets above will unlock full sign-off.
