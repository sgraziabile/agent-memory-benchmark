# Export reading guide — riser assembly extracts

## Why this guide exists

The assembly exports are written by a field extractor that puts one record per line and omits the header
row. The omission is deliberate: the extractor is shared by four teams, and a header written by one team
would be wrong for the others. The order for each family is therefore recorded here and nowhere else, and
every extract names its family on its first line.

## Field order by family

| Extract family | Field order |
|---|---|
| third-riser assembly | asset identifier, certified value, test pressure, verification date |
| north-riser assembly | asset identifier, test pressure, certified value, verification date |
| low-voltage distribution | asset identifier, certified value, verification date |
| roof plant | asset identifier, test pressure, verification date |

The two riser families share a shape but not an order. A record copied from one family into a ticket for
the other parses cleanly and reads wrongly, which is why the family name is printed on the first line of
every extract and why this table is the only place the order is written down.

## Field notes

The certified value is the rating recorded when the assembly is certified and revised only by the
certifying team. The test pressure is a single measurement taken during the pressure test and is not
maintained afterwards. The verification date is the day the reading was checked, not the day the assembly
was installed.

## Corrections

A change to a field order is a new release of this guide and is announced to all four teams together. The
guide is not released per family: one release covers every family, so a family cannot be reading an order
that another family has already changed.
