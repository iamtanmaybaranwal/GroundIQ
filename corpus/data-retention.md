# Data Retention and Deletion

## Retention periods
Application logs are retained for 30 days in hot storage and 12 months in cold archive. Audit logs are retained for seven years to satisfy financial reporting requirements. Customer content is retained for the life of the account. Analytics events are retained for 25 months. Backups are retained for 35 days with point-in-time recovery.

## Deletion requests
A verified customer deletion request must be completed within 30 days. Deletion is propagated to primary storage, search indexes, caches and analytics stores. Backups are not selectively edited; instead the deleted records are suppressed on restore, and the backups themselves age out within 35 days.

## Personal data
Personal data may only be stored in systems listed in the data inventory. Adding a new field that contains personal data requires a privacy review before it ships. Personal data must never be written to application logs; use a stable pseudonymous identifier instead.

## Data residency
EU customer data is stored and processed in the EU region only. Cross-region replication of EU customer content is disabled at the bucket policy level, and a quarterly control test verifies this.

## Access to customer data
Access to production customer data requires an approved just-in-time access grant that expires after four hours and is recorded in the audit log with a stated reason.
