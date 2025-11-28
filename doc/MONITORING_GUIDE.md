# AI Trader - Monitoring & Dashboard Guide

## Quick Start

### 1. Start the System
```bash
python -m uvicorn src.main:app --reload --port 8001
```

### 2. Access the Dashboard
Open your browser and navigate to:
```
http://localhost:8001/dashboard
```

You'll see a real-time dashboard with:
- Portfolio value and daily P&L
- Active positions
- Total trades and win rate
- System performance metrics
- Health checks
- Performance charts
- Recent trades table

---

## API Endpoints

### Health Checks

**GET /api/monitoring/health**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "checks": [
    {
      "name": "system_resources",
      "status": "healthy",
      "message": "System resources OK",
      "latency_ms": 5.2,
      "details": {
        "cpu_percent": 45.3,
        "memory_percent": 62.1,
        "disk_percent": 54.2
      }
    },
    {
      "name": "database",
      "status": "healthy",
      "message": "Database OK",
      "latency_ms": 12.5
    },
    {
      "name": "ml_models",
      "status": "degraded",
      "message": "Some ML models unavailable",
      "details": {
        "lstm_available": true,
        "meta_learner_available": false
      }
    }
  ]
}
```

### Metrics

**GET /api/monitoring/metrics**
```json
{
  "info": {
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "uptime_human": "1.0h"
  },
  "metrics": {
    "trade.count": {
      "count": 25,
      "sum": 25,
      "mean": 1.0,
      "min": 1,
      "max": 1,
      "last": 1
    },
    "trade.execution_latency_ms": {
      "count": 25,
      "mean": 123.4,
      "min": 45.2,
      "max": 350.8,
      "last": 98.3
    },
    "portfolio.equity": {
      "last": 10543.21
    },
    "portfolio.pnl_daily": {
      "last": 234.56
    }
  }
}
```

**GET /api/monitoring/metrics/{metric_name}**

Get detailed stats for a specific metric:
```json
{
  "name": "trade.execution_latency_ms",
  "stats": {
    "count": 100,
    "mean": 125.3,
    "min": 45.0,
    "max": 450.2,
    "last": 98.5
  },
  "percentiles": {
    "p50": 110.5,
    "p95": 280.3,
    "p99": 420.1
  }
}
```

**GET /api/monitoring/metrics/{metric_name}/timeseries**

Get time series data (last 100 points):
```json
{
  "name": "portfolio.equity",
  "series": [
    {
      "timestamp": 1706457600.5,
      "value": 10234.56,
      "tags": {}
    },
    ...
  ]
}
```

### WebSocket Updates

**WS /api/monitoring/ws**

Receive real-time metric updates every 2 seconds:
```javascript
const ws = new WebSocket('ws://localhost:8001/api/monitoring/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Metrics update:', data);
  // data.type === 'metrics_update'
  // data.data contains full metrics summary
};
```

---

## Integrating Metrics in Your Code

### Recording Metrics

```python
from monitoring.metrics import record_metric, increment_counter, MetricNames

# Record a trade execution
record_metric(
    MetricNames.TRADE_EXECUTION_LATENCY_MS,
    latency_ms,
    tags={"symbol": "BTCUSDT", "side": "buy"}
)

# Increment trade counter
increment_counter(
    MetricNames.TRADE_COUNT,
    delta=1,
    tags={"symbol": "BTCUSDT"}
)

# Record portfolio metrics
record_metric(MetricNames.PORTFOLIO_EQUITY, current_equity)
record_metric(MetricNames.PORTFOLIO_PNL_DAILY, daily_pnl)
```

### Adding Custom Health Checks

```python
from monitoring.health import get_health_checker, HealthCheckResult, HealthStatus

async def check_my_service() -> HealthCheckResult:
    """Custom health check."""
    start = time.time()
    try:
        # Check your service
        is_healthy = await my_service.ping()
        
        return HealthCheckResult(
            name="my_service",
            status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
            message="Service OK" if is_healthy else "Service down",
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return HealthCheckResult(
            name="my_service",
            status=HealthStatus.UNHEALTHY,
            message=f"Check failed: {e}",
            latency_ms=(time.time() - start) * 1000,
        )

# Register the check
health_checker = get_health_checker()
health_checker.register_check("my_service", check_my_service)
```

### Sending Alerts

```python
from monitoring.alerts import send_alert, AlertSeverity

# Send a critical alert
await send_alert(
    title="Trading System Halted",
    message="Volatility halt triggered. ATR: 15.2%",
    severity=AlertSeverity.CRITICAL,
    tags={"reason": "volatility", "atr_pct": "15.2"}
)

# Send a warning
await send_alert(
    title="Large Loss Detected",
    message=f"Position loss: -5.2% on BTCUSDT",
    severity=AlertSeverity.WARNING,
    tags={"symbol": "BTCUSDT", "loss_pct": "-5.2"}
)
```

---

## Configuring Alerts

Create `.env` file with alert configuration:

```bash
# Telegram Alerts
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Email Alerts
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASSWORD=your_password
EMAIL_TO=alerts@yourdomain.com

# Webhook Alerts
WEBHOOK_ENABLED=true
WEBHOOK_URL=https://your-webhook-url.com/alerts

# Alert throttling (seconds)
ALERT_THROTTLE_SECONDS=300
```

Then configure in code:
```python
from monitoring.alerts import configure_alerts, AlertConfig

config = AlertConfig(
    telegram_enabled=True,
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    email_enabled=True,
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    smtp_user=os.getenv("SMTP_USER"),
    smtp_password=os.getenv("SMTP_PASSWORD"),
    email_to=["alerts@example.com"],
    throttle_seconds=300,
)

configure_alerts(config)
```

---

## Production Deployment Checklist

### Pre-Deployment
- [ ] Review all health checks are registered
- [ ] Configure alert channels (Telegram/Email)
- [ ] Set up metrics retention (default 1 hour)
- [ ] Test dashboard loads properly
- [ ] Verify WebSocket connections work
- [ ] Check all API endpoints respond

### Deployment
- [ ] Set `APP_ENV=production`
- [ ] Enable HTTPS for dashboard
- [ ] Configure reverse proxy (nginx/caddy)
- [ ] Set up log aggregation (ELK/Datadog)
- [ ] Enable system resource monitoring
- [ ] Configure firewall rules

### Post-Deployment
- [ ] Verify dashboard shows live data
- [ ] Test alert delivery (send test alert)
- [ ] Monitor for 24 hours
- [ ] Review metric accuracy
- [ ] Check for memory leaks
- [ ] Validate health check frequency

---

## Monitoring Best Practices

### 1. Key Metrics to Track
- **Trade Execution Latency**: Should be <200ms p95
- **Portfolio Equity**: Track continuously
- **Win Rate**: Should be >50% for profitable system
- **Position Count**: Monitor for overexposure
- **ML Inference Time**: Should be <50ms p95
- **API Error Rate**: Should be <1%

### 2. Alert Thresholds
```python
# Critical: Immediate action required
- Daily loss >5%
- System crash
- Database connection loss
- All ML models failed

# Warning: Monitor closely
- Daily loss >2%
- ML model degraded
- High latency (>500ms)
- Correlation risk high

# Info: FYI only
- Trade executed
- Position opened
- Daily summary
```

### 3. Dashboard Usage
- Check dashboard every 2-4 hours during trading
- Review performance charts daily
- Monitor health checks continuously
- Investigate any "degraded" status immediately

### 4. Performance Optimization
- Keep metric retention <1 hour in production
- Use WebSocket for real-time updates
- Aggregate metrics in 5-minute buckets
- Clean old metric points regularly

---

## Troubleshooting

### Dashboard Not Loading
```bash
# Check server is running
curl http://localhost:8001/ping

# Check static files exist
ls -la static/dashboard.html

# Check logs
tail -f logs/ai_trader.log
```

### No Metrics Showing
```python
# Verify metrics are being recorded
from monitoring.metrics import get_metrics_collector
collector = get_metrics_collector()
print(collector.get_all_metrics())
```

### WebSocket Disconnects
- Check firewall allows WebSocket connections
- Verify reverse proxy WebSocket config
- Check server not timing out connections
- Monitor for event loop blocking

### Health Checks Failing
```python
# Run individual check
from monitoring.health import get_health_checker
checker = get_health_checker()
result = await checker.run_check("database")
print(result.to_dict())
```

---

## Advanced: Custom Dashboard

Create your own dashboard by using the API:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Custom Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>My Trading Dashboard</h1>
    <div id="equity">Loading...</div>
    <canvas id="chart"></canvas>
    
    <script>
        // Fetch metrics
        async function updateMetrics() {
            const response = await fetch('/api/monitoring/metrics');
            const data = await response.json();
            
            const equity = data.metrics['portfolio.equity']?.last || 0;
            document.getElementById('equity').textContent = 
                `Portfolio: $${equity.toLocaleString()}`;
        }
        
        // Real-time updates via WebSocket
        const ws = new WebSocket('ws://localhost:8001/api/monitoring/ws');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateChart(data.data.metrics);
        };
        
        setInterval(updateMetrics, 5000);
        updateMetrics();
    </script>
</body>
</html>
```

---

## Support

For issues or questions:
- GitHub Issues: https://github.com/m34959203/ai_trader/issues
- Documentation: /docs/PRODUCTION_AUDIT_2025.md
- API Docs: http://localhost:8001/docs

---

**Happy Trading! 🚀**
