# Dataset provenance

- Source: Washington State Open Data Portal (data.wa.gov), dataset `3d5d-sdqb`,
  "Electric Vehicle Population Size History By County".
- Query (retrieved 2026-11-20): rows where state = 'WA' and
  vehicle_primary_use = 'Passenger'; fields date, county,
  electric_vehicle_ev_total; ordered by date, county.
- File: `ev_population_by_county_raw.csv` — 4,602 data rows, monthly,
  starting 2017-01-31, 39 Washington counties.
- `electric_vehicle_ev_total` is the count of registered battery-electric +
  plug-in-hybrid vehicles in the county at month end (a stock, not a flow).
- Licence: Washington State open data (public domain / open government data).
