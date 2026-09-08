# PostgreSQL Runbook

## Topology
Each production service uses a primary PostgreSQL instance with a synchronous standby in a second availability zone and one asynchronous read replica. Connections are pooled through PgBouncer in transaction mode; applications must not hold a connection open across user think-time.

## Failover
Automatic failover promotes the synchronous standby when the primary fails health checks for 30 seconds. Expected downtime during failover is under 60 seconds. After a failover, confirm that the old primary has been fenced before it is rebuilt as a standby, otherwise a split-brain write is possible.

## Backups and recovery
Full backups run nightly, with WAL archiving enabled for point-in-time recovery to any second within the last 35 days. A restore drill is performed every quarter and the measured restore time is recorded; the current restore time objective for a 500 GB database is 45 minutes.

## Common errors
Error `RDS-4471` means the connection pool is exhausted. Check for long-running transactions with `pg_stat_activity`, terminate any transaction idle in transaction for more than five minutes, and confirm the application is releasing connections. Raising `max_connections` is the last resort, not the first response, because each connection costs memory on the primary.

## Migrations
Schema migrations run through the migration runner and must be backward compatible with the previous release. Adding a column is safe; dropping one requires the expand-and-contract pattern across two releases. Never run a migration that takes an exclusive lock on a large table during business hours; use `CREATE INDEX CONCURRENTLY` instead.
