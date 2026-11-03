# Addendum 1 to the Northwind Datacenter Access Policy

Published: 2026-09-14
Effective: immediately
Owner: Change Advisory Board

## A. Supersession notice

Section 3.2 is superseded: production-tier reboots now require Level-2
approval from the Change Advisory Board. Level-1 approval from the on-call
lead is no longer sufficient on its own.

Revision 1 of the access policy remains valid everywhere else, including the
roles table in section 2 and the break-glass rules in section 5.

## B. Reason for the change

Three production-tier reboots in the second quarter were approved by a single
on-call lead and executed outside the maintenance slot recorded on the change
ticket. The Change Advisory Board decided that a second, independent approval
is required for any restart of a production-tier service.

## C. Operational impact

Every production-tier reboot now needs two approvals on the change ticket:
one from the on-call lead and one from the Change Advisory Board. The reboot
window is unchanged.

## D. Compliance window

Teams have until 2026-10-01 to update their runbooks. After that date an
audit finding is raised for every reboot ticket that carries only a Level-1
approval.
