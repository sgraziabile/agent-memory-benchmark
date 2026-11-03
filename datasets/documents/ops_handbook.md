# Northwind Operations Handbook: Service Ports and Endpoints

This handbook lists the listening ports for every internal Northwind service.
Ports are allocated in blocks: telemetry services use 74xx, control-plane
services use 84xx, and storage services use 94xx. Do not reuse a port across
environments; the service registry enforces uniqueness per environment.

## 1. Telemetry pipeline

The log shipper listens on port 7411. The metrics collector listens on port
7422. The trace aggregator listens on port 7433. The ingestion gateway for the
telemetry pipeline listens on port 7443. The retention compactor listens on
port 7455.

Each telemetry service exposes a health endpoint on `/healthz`, which returns
`200 OK` once the service has replayed its local write-ahead log.

## 2. Control plane

The scheduler listens on port 8411. The admission controller listens on port
8422. The policy engine listens on port 8433. The audit relay listens on port
8444. The certificate rotation agent listens on port 8455.

The control plane tolerates a single broker failure. A second failure in the
same availability zone triggers a page to the on-call rotation.

## 3. Storage services

The block store listens on port 9411. The object store listens on port 9422.
The cold archive endpoint listens on port 9433. The replica synchronizer
listens on port 9444.

Cold archives are written once and read rarely. Retrieval requests are queued
and may take up to four hours to complete.

## 4. Facility and cooling

The Rosario facility hosts rows R1 through R8. Row R3 holds the cold-storage
medical biobank chamber, which must stay at minus eighty Celsius. Row R6 holds
the network core. The Buenos Aires facility holds only disaster-recovery
replicas and no primary workloads.

Facility alarms route to the facilities mailbox. Environmental alarms above
30 degrees Celsius escalate to the on-call rotation automatically.

## 5. Escalation contacts

The escalation lead for the Rosario facility is Elena Rostova. The escalation
lead for the Buenos Aires facility is Mateo Ferreira. The failover cluster
identifier for the disaster-recovery pair is `dr-atlas-04`.

Escalation contacts rotate every quarter. The on-call schedule is published on
the internal status page and is authoritative over this handbook.

## 6. Backup windows

Full backups run on Sunday between 01:00 and 04:00 local time. Incremental
backups run daily between 02:00 and 03:00. A restore drill is executed on the
first Monday of every month and its result is recorded in the audit log.

Backup jobs that overrun their window are killed rather than allowed to
encroach on the morning batch jobs.

## 7. Incident severity

Severity 1 incidents are customer-visible outages and require a page within
five minutes. Severity 2 incidents are degraded service and require a page
within thirty minutes. Severity 3 incidents are handled during business hours.

Every Severity 1 incident produces a written post-incident review within five
business days.

## 8. Maintenance slots

Standard maintenance runs Tuesday and Thursday between 03:00 and 05:00 local
time. Emergency maintenance may be scheduled at any time with a Level-3
approval. Rebooting a production-tier service is not a maintenance slot; it is
a change and follows the access policy.
