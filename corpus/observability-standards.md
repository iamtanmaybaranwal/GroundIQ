# Observability Standards

## The three signals
Every service emits structured JSON logs, RED metrics (rate, errors, duration) and distributed traces. Logs carry a correlation id that is propagated across service boundaries in the `X-Correlation-ID` header, so one request can be reconstructed end to end.

## Logging rules
Log at INFO for state changes and at ERROR only for conditions that need human attention; anything logged at ERROR should be actionable. Never log secrets, access tokens or personal data. High-cardinality identifiers belong in structured fields, not in the message string, so they remain queryable.

## Metrics
Latency is reported as a histogram, never as an average, because averages hide the tail that customers actually experience. Dashboards show p50, p95 and p99. Counters are named `<service>_<thing>_total` and always carry a `status` label.

## SLOs and error budgets
Each tier-1 service defines an availability SLO and a latency SLO over a 28-day rolling window. The error budget is the allowed unavailability. When more than half the budget is consumed, the team reviews reliability work at the next planning session. When the budget is exhausted, feature deploys pause until the service is back inside its objective.

## Tracing
Sample 100% of failed requests and 1% of successful ones. Every outbound call records the downstream service, the operation and the status, so a latency regression can be attributed to a specific dependency rather than guessed at.
