# Deployment Policy

## Release windows
Production deployments are permitted Monday through Thursday between 09:00 and 16:00 in the owning team's local timezone. Friday deployments require written approval from the service owner and a named engineer on standby for the following 24 hours. There is a full deployment freeze from 20 December to 2 January and during any active Sev1 incident.

## Progressive delivery
Every production release must roll out progressively. The standard canary schedule is 5% of traffic for 15 minutes, then 25% for 15 minutes, then 100%. The canary is evaluated automatically against the service error rate and p95 latency; a canary that exceeds a 2% error rate or doubles baseline p95 latency is halted and rolled back without human intervention.

## Rollback procedure
To roll back a release, run `shipctl rollback <service> --to <previous-release-id>`. Rollback is always preferred over rolling forward with a hotfix: restore service first, diagnose second. The target rollback time is under five minutes from decision to full traffic restoration. Database migrations must be backward compatible for at least one release so that a rollback never requires a schema revert; use the expand-and-contract pattern for column changes.

## Approvals
Changes to services tagged `tier-1` require a second reviewer and a completed pre-deploy checklist. Configuration-only changes behind an existing feature flag do not need a second reviewer, but the flag change is recorded in the audit log.

## Feature flags
New functionality ships dark behind a feature flag by default. Flags older than 90 days are reported weekly and must be either removed or promoted to permanent configuration, because stale flags are a common cause of untested code paths reaching production.
