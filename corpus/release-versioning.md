# API Versioning and Deprecation

## Versioning scheme
The public API is versioned in the URL path as `/v1`, `/v2` and so on. A new major version is created only for a breaking change. Additive changes such as a new optional field or a new endpoint ship inside the current major version and never require a client change.

## What counts as breaking
Removing or renaming a field, changing a field type, making an optional request field required, changing an error code, or tightening validation are all breaking changes. Adding a new field to a response is not breaking, so clients must ignore unknown fields.

## Deprecation policy
A deprecated API version is supported for a minimum of 12 months after the deprecation notice. Deprecated endpoints return a `Sunset` header with the removal date and a `Deprecation` header set to true. Customers with recent traffic on a deprecated endpoint are contacted directly at 6 months, 3 months and 1 month before removal.

## Internal service versioning
Internal services follow semantic versioning for their client libraries. Consumers pin a major version and upgrade minor versions freely. Any breaking change to an internal contract requires the provider to support both shapes until every consumer has migrated, which is verified from the request logs rather than assumed.

## Release notes
Every release publishes notes describing user-visible changes, migration steps, and any change to rate limits or error codes.
