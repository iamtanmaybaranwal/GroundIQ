# API Rate Limits

## Tiers
The public API is rate limited per API key. The Free tier allows 60 requests per minute with a burst of 20. The Pro tier allows 600 requests per minute with a burst of 100. The Enterprise tier allows 6,000 requests per minute with a burst of 1,000 and supports negotiated custom limits.

## Algorithm
Limits are enforced with a token bucket refilled continuously at the tier steady rate, so short bursts are absorbed while the long-run average is capped. Limits are shared across all replicas of the gateway through a central store, meaning a client cannot gain extra capacity by spreading traffic across regions.

## Response headers
Every response carries `X-RateLimit-Limit`, `X-RateLimit-Remaining` and `X-RateLimit-Reset`. When a client exceeds its limit the API returns HTTP 429 with a `Retry-After` header in seconds. Clients must honour `Retry-After` and use exponential backoff with jitter; clients that retry immediately and repeatedly may have their key suspended.

## Endpoint-specific limits
Search endpoints are limited to one third of the account overall quota because they are significantly more expensive to serve. Bulk export endpoints are limited to 10 requests per hour regardless of tier.

## Requesting an increase
Ask through the support portal with the account id, the endpoint, the requested limit and the business justification. Rate limit increases are reviewed within two business days and are granted for a fixed period with an automatic review.
