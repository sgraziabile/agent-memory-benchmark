# Northwind Datacenter Access Policy (Revision 1)

Document owner: Platform Security
Applies to: `prod`, `staging` and `lab` environments
Classification: Internal

## 1. Scope

This policy governs human and automated access to every Northwind facility,
including the Rosario primary datacenter and the Buenos Aires disaster
recovery site. It covers change management, customer-data handling, and the
emergency accounts used by the platform on-call rotation.

Nothing in this document overrides a contractual data-processing agreement.
Where a customer contract is stricter, the contract wins.

## 2. Roles and authorization levels

Every request must carry one of the following authorization levels:

| Role                        | Authorization level | Notes                                  |
|-----------------------------|---------------------|----------------------------------------|
| Platform engineer           | Level-0             | Read-only in `prod`                    |
| Senior platform engineer    | Level-1             | May approve standard changes           |
| Change Advisory Board (CAB) | Level-2             | Owns the change calendar; §4 data access |
| Security incident commander | Level-3             | Break-glass and data-erasure decisions |

An approval is bound to a single change ticket. Reusing an approval for a
second ticket is treated as an audit finding, not as an approval.

## 3. Change management

### 3.1 Standard changes

Standard changes (dashboards, alerts, log retention tweaks) may be applied
during business hours after a Level-0 review and a green CI pipeline.

### 3.2 Production-tier reboots

Production-tier reboots require Level-1 approval from the on-call lead. The
approval must be recorded on the change ticket before the reboot starts, and
the reboot window is limited to the maintenance slot named on that ticket.

### 3.3 Emergency changes

An emergency change may bypass the normal window but never the approval
requirement. If the approver cannot be reached, the Security incident
commander assumes approval authority and records the delegation in the
incident log.

## 4. Access to customer data

Customer data may only be read from the `prod` environment by a Level-2
holder, and only through the audited query gateway. Bulk exports require a
Level-3 approval and expire after 24 hours.

## 5. Break-glass accounts

The break-glass emergency account identifier is `iris-breakglass-77`. The
credential is sealed in the hardware token safe at the Rosario facility and
is released only to the Security incident commander.

Every break-glass use must be followed by a post-incident review within five
business days. Accounts that are not rotated within that window are disabled.

## 6. Audit and review

Access reviews run quarterly. The Platform Security team samples 10% of
Level-1 and Level-2 approvals and reconciles them against change tickets.
Findings are reported to the Change Advisory Board.

## 7. Exceptions

Exceptions require a written justification, a named owner and an expiry date
no more than 90 days in the future. Expired exceptions are closed
automatically by the compliance job.
