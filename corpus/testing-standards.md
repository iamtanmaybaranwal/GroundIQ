# Testing Standards

## The pyramid
Every service maintains fast unit tests, a smaller layer of integration tests that exercise real dependencies such as a database or a message broker, and a thin layer of end-to-end tests covering only critical user journeys. End-to-end tests are the most expensive and the most flaky, so they cover journeys, not features.

## Coverage
Line coverage is a signal, not a target. Pull requests must not reduce coverage on the files they touch, and any new branch of business logic needs a test. Chasing a global coverage percentage produces tests that assert implementation details and make refactoring harder.

## Flaky tests
A test that fails intermittently is quarantined within one working day and either fixed or deleted within one week. A quarantined test does not block the pipeline but is reported daily. Retrying a flaky test until it passes is prohibited, because it hides real race conditions.

## Test data
Tests must not depend on shared mutable state or on the order in which they run. Each test creates the data it needs and cleans up after itself. Production data is never copied into a test environment; use generated fixtures instead.

## Pre-merge checks
The pipeline runs unit tests, linting, type checking and a security scan on every pull request. Integration tests run on merge to the main branch, and the full end-to-end suite runs before a production release.
