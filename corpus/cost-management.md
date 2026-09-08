# Cloud Cost Management

## Ownership and tagging
Every cloud resource must carry `owner`, `service` and `environment` tags. Untagged resources are reported daily and are automatically deleted after 14 days in non-production accounts. Each team sees its own spend in the weekly cost report and owns its budget.

## Budgets and alerts
Each team sets a monthly budget. Alerts fire at 80% and 100% of budget, and a forecast alert fires when projected month-end spend exceeds the budget by more than 10%. Budget breaches are discussed in the monthly engineering review, not escalated as incidents.

## Rightsizing
Instances running below 20% average CPU for 14 days are flagged for rightsizing. Non-production environments shut down automatically outside 08:00 to 20:00 on weekdays, which typically removes about 60% of non-production compute cost.

## Commitments
Baseline steady-state compute is covered by one-year savings plans; burst capacity runs on-demand. Commitment coverage is reviewed quarterly, targeting 70% coverage of the trailing baseline so that a workload change never leaves the company paying for unused commitments.

## Storage
Object storage lifecycle rules move data to infrequent-access after 30 days and to archive after 180 days. Snapshots older than 90 days are deleted unless tagged for retention, since orphaned snapshots are the most common source of silent storage growth.
