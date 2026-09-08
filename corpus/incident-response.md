# Incident Response

## Severity levels
Sev1 means a complete outage or data loss affecting all customers. Sev2 means a major feature is unavailable or a subset of customers is severely degraded. Sev3 means a minor degradation with a workaround. Sev4 is a cosmetic or internal-only issue.

## Response targets
Sev1 requires acknowledgement within 5 minutes and a status page update within 15 minutes. Sev2 requires acknowledgement within 15 minutes. Sev3 is handled during business hours. The incident commander is the first responder to the page unless they explicitly hand the role over.

## During an incident
Open a dedicated incident channel named `inc-<date>-<short-name>` and keep all coordination there. The incident commander does not debug; they coordinate, delegate and communicate. Post a customer-facing status page update every 30 minutes for a Sev1, even when the update is that the team is still investigating.

## Escalation
If a Sev1 is not mitigated within 30 minutes, page the secondary on-call and the engineering manager for the owning service. If it is not mitigated within 60 minutes, page the director of engineering. Escalating early is never penalised.

## Postmortems
Every Sev1 and Sev2 requires a written postmortem within five business days. Postmortems are blameless: they describe systems and contributing factors, never individuals. Each postmortem must list action items with a named owner and a due date, and those action items are tracked to completion in the normal sprint backlog.
