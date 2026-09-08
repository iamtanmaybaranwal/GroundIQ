# Security and Access Control

## Authentication
All internal systems authenticate through the company SSO provider. Hardware-backed multi-factor authentication is mandatory for every employee and contractor. Shared accounts are prohibited; where a system cannot support individual accounts, access is brokered through a bastion that records the individual identity.

## Least privilege
Access is granted by role, not by individual, and defaults to read-only. Write access to production requires just-in-time elevation that expires automatically after four hours. Quarterly access reviews remove entitlements that have not been used in 90 days.

## Secrets management
Secrets live in the central secrets manager and are injected at runtime. Secrets must never be committed to a repository, pasted into a ticket, or written into a container image. Application secrets are rotated every 90 days and database credentials every 30 days. Any secret exposed in a commit is treated as compromised and rotated immediately, even if the commit was force-pushed away.

## Key management
Encryption keys are managed by the cloud KMS with automatic annual rotation. Customer data is encrypted at rest with AES-256 and in transit with TLS 1.3. Downgrading a service to TLS 1.2 requires a documented exception with an expiry date.

## Vulnerability response
Critical vulnerabilities must be patched within 7 days, high severity within 30 days, and medium severity within 90 days. The clock starts when the advisory is published, not when the ticket is triaged.
