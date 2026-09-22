# Where Bus Riders Go — Sun Metro Bus Stop Mobility

An interactive dashboard answering two questions for El Paso's Sun Metro bus
system: **when people ride the bus, what streets do they walk, and where do
they actually go** — home, work, or a specific errand?

**Live dashboard:** `https://hoffmanap.github.io/sunmetro/dashboard/`

---

## Data sources

| Source | What it provides | Access |
|---|---|---|
| **Sun Metro / City of El Paso GIS** | Bus stop locations, routes served, addresses | `gis.elpasotexas.gov` ArcGIS REST layer, fetched live |
| **Azira/Ubermedia device pings** | Anonymized, hashed mobile device location pings (raw pathing) and each device's own resolved "common evening location" (home) and "common daytime location" (work) | Commercial data license — **not included in this repo** |
| **SafeGraph Places** | Points of interest (name, category, location) for identifying errand destinations | `el_paso_all.geojson` extract — **not included in this repo** |
| **OpenStreetMap** | Named street geometry, for snapping walked paths to real streets | Overpass API, fetched live |
| **U.S. Census TIGERweb** | ZIP Code Tabulation Area (ZCTA) boundaries, for the home-zip filter | TIGERweb REST API, fetched live and cached |

**Every device is identified only by an irreversible hash** in the source
data — there are no names, phone numbers, or addresses tied to a person.
Even so, raw location pings are inherently sensitive, so **none of the raw
Azira/SafeGraph exports are committed to this repository.** Only the
aggregated dashboard data (`dashboard/data/`) is public — see
[Privacy](#privacy--what-is-and-isnt-public) below for exactly what that
contains.

---

## Methodology, step by step

1. **Fetch and buffer bus stops** — pull every active Sun Metro stop,
   dedupe stops that share a corner (many routes share the same physical
   stop), and draw a small catchment buffer (50m) around each one.
2. **Extract device pings** — query the Azira platform using the buffer
   polygons, pulling every device ping that fell inside a stop's catchment,
   plus each device's own resolved home/work location.
3. **Identify riders** — a device is only counted as a likely rider if it
   *dwelled* at a stop for a sustained period (not just a single passing
   ping, which is usually a car or pedestrian walking by on the adjacent
   street).
4. **Classify how they left the stop** — walking, driving, or biking, based
   on speed between consecutive pings. A known GPS quirk (a car
   accelerating from a stop briefly reads at bike-range speed) is corrected
   for.
5. **Find where walkers actually arrived** — rather than assuming "wherever
   the last ping happened to be" is the destination, the pipeline looks for
   an actual **arrival dwell**: a cluster of pings that stayed in one place
   for a sustained period. That point is checked against the device's own
   home/work location, and against SafeGraph Places, to resolve a
   destination type.
6. **Snap the walked path to real streets** — using OpenStreetMap road
   geometry, so "what streets do people walk" has real street names, not
   just raw GPS coordinates.
7. **Aggregate for the dashboard** — individual trips are pooled into a
   compact dataset (`build_dashboard_data.py`) that the dashboard filters
   entirely in the browser.

## Known limitations

- **~36% of walking trips have no usable pings after leaving the stop.**
  This is a real limit of how often the location SDK reports a device's
  position, not a bug — a trip with no further pings simply can't be
  traced to a destination.
- **A handful of "destinations" are Sun Metro's own facilities**
  (e.g. maintenance/operations yards), most likely reflecting employees,
  not riders on errands. These should be filtered or footnoted in any
  published analysis.
- **A few SafeGraph sub-places (ATMs, kiosks) remain separate from their
  host store** when SafeGraph itself doesn't tag a parent venue for that
  specific record — most are correctly merged, some aren't.
- **Street-name matching covers roughly 95%** of walked trips at the
  current snap radius (45m); the remainder are typically unnamed ways
  (parking lots, private drives) that don't exist as named streets in
  OpenStreetMap.

---

## Privacy — what is and isn't public

The dashboard **never shows an individual person's home or work location
as a point.** Home and work are only ever shown as:
- a **pooled hex cell** (~174m across) with a rider count, or
- a **zip code** aggregate.

No device ID, exact home/work coordinate, or individual trip-to-trip
linkage is exposed in `dashboard/data/`. Raw pings and SafeGraph exports
(under `pulls/*/raw/`) are excluded from version control via `.gitignore`
and must never be committed.

---

## Repo structure

```
.
├── index.html                       ← redirects to dashboard/ (for GitHub Pages)
├── README.md                        ← this file
│
├── fetch_and_buffer_bus_stops.py    ← pipeline step 1
├── split_bus_stop_buffers_for_extract.py   ← pipeline step 2 (splits extract by area cap)
├── analyze_bus_rider_home_work.py   ← pipeline step 3 (identifies riders, home/work)
├── trace_departures_and_destinations.py  ← pipeline step 4 (mode + destination)
├── aggregate_walking_paths.py       ← pipeline step 5 (street/hex aggregation, static map)
├── build_dashboard_data.py          ← turns pipeline output into dashboard/data/<pull-id>/
│
├── dashboard/
│   ├── index.html                   ← the interactive app
│   └── data/
│       ├── manifest.json            ← lists every pull, powers the pull selector
│       └── <pull-id>/
│           ├── trips.json
│           ├── stops.json
│           ├── routes.json
│           ├── categories.json
│           ├── streets.geojson
│           └── zips.geojson
│
└── pulls/
    └── <pull-id>/
        ├── raw/                     ← Azira/SafeGraph exports -- gitignored, never committed
        └── (bus_stop_output/, rider_output/, trip_output*/, street_output*/)
```

## Running the pipeline for a new data pull

```bash
python fetch_and_buffer_bus_stops.py --out pulls/<id>/bus_stop_output/
python split_bus_stop_buffers_for_extract.py --by-stop pulls/<id>/bus_stop_output/bus_stop_buffers_by_stop.geojson --out pulls/<id>/bus_stop_output/extract_groups/
# (pull device data from Azira using the extract group polygons)
python analyze_bus_rider_home_work.py --stops pulls/<id>/bus_stop_output/bus_stops_clustered.csv --home <cel.tsv.gz> --work <cdl.tsv.gz> --pathing <pathing.zip> --out pulls/<id>/rider_output/
python trace_departures_and_destinations.py --riders pulls/<id>/rider_output/rider_stop_assignments.csv --pathing <pathing.zip> --safegraph-places <safegraph.geojson> --out pulls/<id>/trip_output/
python build_dashboard_data.py --pull-id <id> --riders pulls/<id>/rider_output/rider_stop_assignments.csv --trips pulls/<id>/trip_output/trip_destinations.csv --vehicle pulls/<id>/trip_output/vehicle_departures.csv --walking-paths pulls/<id>/trip_output/walking_paths.geojson --stops pulls/<id>/bus_stop_output/bus_stops_clustered.csv --out dashboard/data/
```

The dashboard's pull selector picks up the new pull automatically — nothing
else needs to change.

## Viewing locally

```bash
cd dashboard
python -m http.server 8000
```

Then open `http://localhost:8000`. Opening `index.html` directly
(`file://…`) will not work — browsers block the `fetch()` calls the
dashboard uses to load its data without a real server.
