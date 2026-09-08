# Kubernetes Standards

## Namespaces and ownership
Every service runs in a namespace named after its team and service, and each namespace carries an owner label used for cost allocation and paging. Workloads without an owner label are blocked by the admission controller.

## Resource requests and limits
Every container must declare CPU and memory requests. Memory limits are mandatory and must equal the request, because a container that exceeds its memory limit is killed and unbounded memory is the most common cause of noisy-neighbour incidents. CPU limits are discouraged: CPU is compressible, and a hard CPU limit causes throttling that shows up as unexplained latency.

## Health probes
Every workload defines a readiness probe and a liveness probe, and they must not be the same endpoint. Readiness reflects whether the pod can serve traffic right now; liveness reflects whether the process is unrecoverable. A liveness probe that checks a downstream dependency will cascade an outage across the fleet, so probes must only check the local process.

## Autoscaling
Horizontal pod autoscaling targets 70% CPU utilisation with a minimum of three replicas for tier-1 services, so that losing one node never removes a majority. Scale-down is stabilised over five minutes to avoid thrashing.

## Deployments
Rolling updates use `maxUnavailable: 0` and `maxSurge: 1`. Every workload defines a pod disruption budget so that node drains during cluster upgrades cannot take a service below quorum.
