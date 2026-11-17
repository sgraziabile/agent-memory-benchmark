# Asset registry — Viaduct facility (excerpt)

The registry is the index of assets at the Viaduct facility. It answers one question: which asset am I
looking at, and what is its identifier? Recorded values are deliberately excluded — each row names the
document where the value is held, so that a value has exactly one home and correcting it does not require
correcting an index.

The excerpt below covers the cooling and air-handling assets of the north half of the facility. The full
registry, including the low-voltage distribution and the roof plant, lives in the same series with a
different suffix and is reissued whenever an asset is added or retired.

## Cooling assets

| Asset | Description | Value held in |
|---|---|---|
| NBL-420 | West-aisle cooling loop | limit schedule |
| NBL-118 | East-aisle cooling loop | limit schedule |
| NBL-501 | South-aisle cooling loop | limit schedule |
| TRB-260 | North cooling loop | limit schedule |
| TRB-318 | Return manifold, north half | limit schedule |
| TRB-515 | Return manifold, south half | limit schedule |
| NBL-604 | Make-up water loop | limit schedule |

## Air handling assets

| Asset | Description | Value held in |
|---|---|---|
| AHU-207 | Air-handling unit, row 3 | limit schedule |
| AHU-233 | Air-handling unit, row 5 | limit schedule |
| AHU-305 | Air-handling unit, roof enclosure | limit schedule |
| AHU-388 | Air-handling unit, loading bay | limit schedule |

## Reading the registry

A row's description is written for a person standing in front of the asset, not for a database. Where two
assets could be confused — the two aisle loops that share a manifold, for instance — the description names
the aisle and the registry does not abbreviate it, because an abbreviated description is exactly the kind of
text that a reader completes from memory.

Identifiers are permanent. An identifier is never reused when an asset is retired, so an old identifier
found in a ticket always refers to the same physical unit even if that unit has since been replaced. Where a
replacement kept the position, the new asset receives a new identifier and the registry records the
supersession rather than overwriting the row.

## Corrections

A correction to a description is a registry change and is reviewed by the facilities lead. A correction to a
recorded value is not: values live in the register, and their correction is a register change. Keeping those
two paths separate is why the index can be read quickly during an incident without anyone needing to check
whether the value beside an identifier is current.

## What the registry is not

It is not a maintenance history, a warranty list or a commissioning record. Those documents exist, they cite
identifiers from this registry, and they are not substitutes for it. A ticket that describes an asset without
its identifier is treated as incomplete at the change review, which is the reason the registry is short
enough to be read aloud.
