# Structured Logging & Central Collection

The API now emits JSON-formatted logs with a `trace_id` field that is propagated
from incoming requests (header `X-Trace-Id` or `X-Request-ID`).  To forward the
logs into a central platform (Loki, ELK, etc.) deploy a lightweight collector
as a DaemonSet.  Example Fluent Bit configuration:

```ini
[INPUT]
    Name              tail
    Path              /var/log/containers/*ai-trader*.log
    Parser            docker
    Tag               ai-trader

[FILTER]
    Name              modify
    Match             ai-trader
    Rename            log message

[FILTER]
    Name              nest
    Match             ai-trader
    Operation         lift
    Nested_under      context

[OUTPUT]
    Name              loki
    Match             ai-trader
    host              loki.monitoring.svc
    labels            job=ai-trader,trace_id=$trace_id,level=$level
```

Key fields emitted per log entry:

* `timestamp` (ISO-8601, UTC)
* `level`, `logger`, `message`
* `trace_id` propagated to responses and exception payloads
* `context` (structured extras, masked via `utils.logging_utils`)

Ensure alert pipelines (Alertmanager, Slack, PagerDuty) include the trace ID in
notifications so responders can correlate events with application logs.
