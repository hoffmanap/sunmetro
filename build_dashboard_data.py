#!/usr/bin/env python3
"""
build_dashboard_data.py
=========================
Turns the outputs of the bus-stop mobility pipeline (rider_stop_assignments.
csv, trip_destinations.csv, vehicle_departures.csv, walking_paths.geojson,
bus_stops_clustered.csv) into a compact set of JSON/GeoJSON files the
interactive dashboard (index.html) loads and cross-filters entirely in the
browser -- no server-side recomputation needed once this has run.

ONE PULL, ONE FOLDER. Each run writes into data/<pull-id>/ and adds an
entry to data/manifest.json. The dashboard reads manifest.json to build its
pull selector, so adding a future data pull is just: run this script again
with a new --pull-id, point it at that pull's pipeline outputs, and the
dashboard picks it up automatically -- nothing else to wire up.

OUTPUTS (per pull, under data/<pull-id>/)
  trips.json       -- one compact record per rider-stop episode: mode
                       (walk/car/bike), resolved destination (home/work/
                       errand, with name+category+coords for errands),
                       home/work hex (COARSE resolution -- individual home
                       locations are never exposed, only which pooled area
                       cell a rider's home falls in, matching the privacy
                       stance already established in this pipeline), and
                       for walking trips: the FINE-resolution hex cells the
                       walked path passed through, plus the snapped street
                       name. This one file is what every filter/chart in
                       the dashboard operates on.
  stops.json        -- stop_id, lat/lon, routes served, address, rider count
  routes.json        -- route -> stop_ids it serves, total riders
  categories.json     -- distinct SafeGraph top_category values + counts,
                       for the category filter buttons
  zips.geojson        -- ZCTA (zip code) boundaries clipped to the data's
                       bounding box, fetched from Census TIGERweb (best-
                       effort -- cached after the first successful fetch so
                       later pulls in the same area don't re-fetch)

USAGE
  python build_dashboard_data.py \\
      --pull-id 10169459 \\
      --riders rider_output/rider_stop_assignments.csv \\
      --trips trip_output/trip_destinations.csv \\
      --vehicle trip_output/vehicle_departures.csv \\
      --walking-paths trip_output/walking_paths.geojson \\
      --stops bus_stops_clustered.csv \\
      --out data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import requests

import aggregate_walking_paths as awp  # reuses the tested OSM street-snapping logic

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_dashboard_data")

DEFAULT_FINE_HEX_RESOLUTION = 12   # ~9.4m edge -- street-scale, for path visualization
DEFAULT_COARSE_HEX_RESOLUTION = 9  # ~174m edge -- privacy-pooled home/work areas
MODE_CODE = {"walking": 0, "car": 1, "biking": 2}
DEST_TYPE_CODE = {"home": 0, "work": 1, "errand": 2, "unresolved": 3}


def build_fine_hex_path(coords: list, resolution: int) -> list:
    """coords: [[lon, lat], ...] from a walking_paths.geojson LineString.
    Returns deduped, order-preserving list of H3 cell ids the path passed
    through, at a resolution fine enough to actually distinguish streets."""
    cells = []
    seen = set()
    for lon, lat in coords:
        cell = h3.latlng_to_cell(lat, lon, resolution)
        if cell not in seen:
            cells.append(cell)
            seen.add(cell)
    return cells


def fetch_zcta_boundaries(bbox: tuple, cache_path: Path) -> dict | None:
    """bbox: (min_lon, min_lat, max_lon, max_lat). Uses Census TIGERweb's
    public REST API (no key required) to fetch ZCTA (zip code) polygons
    intersecting the bounding box. Cached to cache_path after a successful
    fetch so re-running for a pull in the same area doesn't re-fetch.

    Layer: TIGERweb/PUMA_TAD_TAZ_UGA_ZCTA/MapServer/11 ("2020 Census ZIP
    Code Tabulation Areas", field ZCTA5) -- an earlier version of this
    function pointed at Tracts_Blocks/MapServer/2, which is actually the
    2020 Census Blocks layer and has no ZCTA5 field at all; it was failing
    silently every time (returning nothing, so the dashboard's zip filter
    just looked broken). Verified live against a real El Paso bbox before
    shipping this fix -- 17 real ZCTA polygons with correct ZCTA5 values.

    A metro-wide bbox (riders' homes can be scattered across the whole
    area, not just near stops) is a big enough query that TIGERweb has been
    seen to time out at 60s under load -- retries with a longer timeout
    before giving up, same pattern already used for the Overpass street
    fetch."""
    if cache_path.exists():
        log.info("  using cached zip boundaries at %s", cache_path)
        return json.load(open(cache_path))
    min_lon, min_lat, max_lon, max_lat = bbox
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/PUMA_TAD_TAZ_UGA_ZCTA/MapServer/11/query"
    params = {
        "geometry": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "geometryType": "esriGeometryEnvelope", "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outFields": "ZCTA5,BASENAME", "outSR": "4326", "f": "geojson",
    }
    last_error = None
    for attempt, timeout in enumerate([60, 120, 180]):
        try:
            log.info("  querying TIGERweb for ZCTA boundaries (attempt %d, timeout %ds) ...", attempt + 1, timeout)
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("features"):
                log.warning("  TIGERweb returned no ZCTA features for this bbox -- zip filter will be unavailable.")
                return None
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(data, open(cache_path, "w"))
            log.info("  fetched %d zip boundaries, cached to %s", len(data["features"]), cache_path)
            return data
        except Exception as e:
            last_error = e
            log.info("    attempt %d failed (%s), %s", attempt + 1, e,
                      "retrying with a longer timeout ..." if attempt < 2 else "giving up.")
    log.warning("  Could not fetch zip boundaries after 3 attempts (%s) -- zip filter will be unavailable this run.",
                last_error)
    return None


def build_zip_lookup(zips_geojson: dict):
    """Returns a function mapping (lat, lon) -> ZCTA5 string or None, using
    shapely for correct point-in-polygon (handles MultiPolygon ZCTAs, which
    are common -- some zip codes are genuinely disjoint)."""
    from shapely.geometry import shape, Point
    entries = [(shape(f["geometry"]), f["properties"].get("ZCTA5")) for f in zips_geojson["features"]]

    def lookup(lat, lon):
        pt = Point(lon, lat)
        for geom, zcta in entries:
            if geom.contains(pt):
                return zcta
        return None
    return lookup


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pull-id", required=True, help="Identifier for this data pull (e.g. an Azira job id). "
                                                       "Used as the output subfolder name and the dashboard's pull selector label.")
    p.add_argument("--riders", required=True, help="rider_stop_assignments.csv")
    p.add_argument("--trips", required=True, help="trip_destinations.csv")
    p.add_argument("--vehicle", required=True, help="vehicle_departures.csv")
    p.add_argument("--walking-paths", required=True, help="walking_paths.geojson")
    p.add_argument("--vehicle-paths", default=None,
                    help="vehicle_paths.geojson from trace_departures_and_destinations.py (optional -- older "
                         "pipeline runs won't have this; car/biking trips just won't get street/path data without it).")
    p.add_argument("--stops", required=True, help="bus_stops_clustered.csv")
    p.add_argument("--dwell-confirmed-only", action="store_true", default=True,
                    help="Only include walking trips with a confirmed arrival dwell (default: on).")
    p.add_argument("--include-unconfirmed", dest="dwell_confirmed_only", action="store_false",
                    help="Include all walking trips regardless of dwell confirmation.")
    p.add_argument("--fine-hex-resolution", type=int, default=DEFAULT_FINE_HEX_RESOLUTION)
    p.add_argument("--coarse-hex-resolution", type=int, default=DEFAULT_COARSE_HEX_RESOLUTION)
    p.add_argument("--max-snap-m", type=float, default=45.0, help="Street-snap radius (see aggregate_walking_paths.py).")
    p.add_argument("--skip-streets", action="store_true", help="Skip OSM street snapping for this run.")
    p.add_argument("--skip-zips", action="store_true", help="Skip fetching zip code boundaries for this run.")
    p.add_argument("--out", default="data/", help="Parent data folder (this pull writes to <out>/<pull-id>/).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out) / args.pull_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading stops from %s ...", args.stops)
    stops = pd.read_csv(args.stops)
    stops["routes"] = stops["routes"].fillna("").apply(lambda s: [r for r in str(s).split("|") if r])

    log.info("Loading riders (home/work lookup) from %s ...", args.riders)
    riders = pd.read_csv(args.riders)
    riders["device_id"] = riders["device_id"].astype(str)
    # home_zip/work_zip are present if analyze_bus_rider_home_work.py was run
    # with an updated version that captures the CEL/CDL export's own postal
    # code field -- a real device-reported zip, not a reverse-geocoded
    # guess, so no external boundary fetch is needed at all for this. Older
    # rider_stop_assignments.csv files won't have these columns; handled
    # gracefully below.
    lookup_cols = ["home_lat", "home_lon", "work_lat", "work_lon"]
    has_reported_zip = "home_zip" in riders.columns
    if has_reported_zip:
        lookup_cols.append("home_zip")
        # Pandas infers an all-numeric-looking zip column as float64, which
        # turns "79936" into 79936.0 and then, when stringified, "79936.0"
        # -- fixed at the STRING level (not by round-tripping through int)
        # so a zip that genuinely has a leading zero elsewhere in the
        # country (e.g. "02139") isn't corrupted by dropping it.
        riders["home_zip"] = riders["home_zip"].astype(str).str.replace(r"\.0$", "", regex=True)
        riders.loc[riders["home_zip"].isin(["nan", "None"]), "home_zip"] = None
        log.info("  rider_stop_assignments.csv has a reported home_zip column -- using it directly.")
    else:
        log.info("  no home_zip column in rider_stop_assignments.csv (older pipeline run) -- "
                  "the zip filter will be unavailable this run. Re-run analyze_bus_rider_home_work.py "
                  "with the current version to get it.")
    riders_lookup = riders.set_index(["device_id", "stop_id"])[lookup_cols]

    log.info("Loading walking trips from %s ...", args.trips)
    trips = pd.read_csv(args.trips)
    trips["device_id"] = trips["device_id"].astype(str)
    trips = trips[trips["departure_mode"] == "walking"]
    if args.dwell_confirmed_only and "dwell_confirmed" in trips.columns:
        before = len(trips)
        trips = trips[trips["dwell_confirmed"] == True]  # noqa: E712
        log.info("  --dwell-confirmed-only: kept %d of %d walking trips.", len(trips), before)

    log.info("Loading vehicle departures from %s ...", args.vehicle)
    try:
        vehicle = pd.read_csv(args.vehicle)
        vehicle["device_id"] = vehicle["device_id"].astype(str)
    except pd.errors.EmptyDataError:
        log.info("  vehicle_departures.csv is empty -- no non-foot departures this pull.")
        vehicle = pd.DataFrame(columns=["device_id", "stop_id", "departure_mode"])

    log.info("Loading walking paths from %s ...", args.walking_paths)
    paths_data = json.load(open(args.walking_paths))
    path_by_key = {
        f"{f['properties']['device_id']}||{f['properties']['stop_id']}": f["geometry"]["coordinates"]
        for f in paths_data["features"]
    }

    # Vehicle (car/biking) departure paths -- optional, since older pipeline
    # runs won't have this file. Merged into the SAME path_by_key lookup so
    # every downstream step (fine hex, street snap) treats them uniformly;
    # only the destination-resolution logic (which never ran for vehicles)
    # differs.
    if args.vehicle_paths:
        log.info("Loading vehicle departure paths from %s ...", args.vehicle_paths)
        vpaths_data = json.load(open(args.vehicle_paths))
        for f in vpaths_data["features"]:
            key = f"{f['properties']['device_id']}||{f['properties']['stop_id']}"
            path_by_key[key] = f["geometry"]["coordinates"]
        log.info("  %d vehicle departure paths loaded.", len(vpaths_data["features"]))

    segments, grid, cell_deg = [], {}, 0.0003
    if not args.skip_streets:
        all_lats = [pt[1] for coords in path_by_key.values() for pt in coords]
        all_lons = [pt[0] for coords in path_by_key.values() for pt in coords]
        if all_lats:
            bbox = (min(all_lons) - 0.01, min(all_lats) - 0.01, max(all_lons) + 0.01, max(all_lats) + 0.01)
            log.info("Fetching OSM streets for street-name snapping ...")
            try:
                segments = awp.fetch_osm_streets(bbox)
                cell_deg = max(args.max_snap_m / 100_000.0, 0.0003)
                grid = awp.build_street_grid(segments, cell_deg)
                log.info("  %d street segments loaded.", len(segments))
            except Exception as e:
                log.warning("  Could not fetch OSM streets (%s) -- trips.json will have no street names this run.", e)

    # Zip code boundary fetch removed as the primary path for the zip
    # filter -- the CEL/CDL export's OWN postal-code field (captured by
    # analyze_bus_rider_home_work.py into home_zip, when present) is a real
    # device-reported zip code, more reliable than reverse-geocoding lat/lon
    # against an externally-fetched boundary service (which needed retries
    # for a slow/flaky Census endpoint and only ever existed to support
    # this one feature). --skip-zips is now a no-op kept for CLI compatibility.

    log.info("Building trip records ...")
    trip_records = []
    category_counts: Counter = Counter()
    for _, r in trips.iterrows():
        key = f"{r['device_id']}||{r['stop_id']}"
        hw = riders_lookup.loc[(r["device_id"], r["stop_id"])] if (r["device_id"], r["stop_id"]) in riders_lookup.index else None
        home_hex = h3.latlng_to_cell(hw["home_lat"], hw["home_lon"], args.coarse_hex_resolution) \
            if hw is not None and pd.notna(hw.get("home_lat")) else None
        work_hex = h3.latlng_to_cell(hw["work_lat"], hw["work_lon"], args.coarse_hex_resolution) \
            if hw is not None and pd.notna(hw.get("work_lat")) else None
        home_zip = hw.get("home_zip") if hw is not None and has_reported_zip else None
        home_zip = None if pd.isna(home_zip) else str(home_zip)

        rec = {"s": r["stop_id"], "m": MODE_CODE["walking"], "dt": DEST_TYPE_CODE.get(r["destination_type"], 3),
               "hh": home_hex, "wh": work_hex, "hz": home_zip}
        if r["destination_type"] == "errand":
            rec["dn"] = r.get("destination_name")
            rec["dc"] = r.get("destination_category")
            rec["dlat"] = r.get("destination_lat")
            rec["dlon"] = r.get("destination_lon")
            if pd.notna(r.get("destination_category")):
                category_counts[r["destination_category"]] += 1

        coords = path_by_key.get(key)
        if coords:
            rec["ph"] = build_fine_hex_path(coords, args.fine_hex_resolution)
            if segments:
                street_votes = Counter()
                for lon, lat in coords:
                    name = awp.snap_point_to_street(lat, lon, segments, grid, cell_deg, args.max_snap_m)
                    if name:
                        street_votes[name] += 1
                if street_votes:
                    rec["st"] = street_votes.most_common(1)[0][0]
        trip_records.append(rec)

    for _, r in vehicle.iterrows():
        key = (r["device_id"], r["stop_id"])
        hw = riders_lookup.loc[key] if key in riders_lookup.index else None
        home_hex = h3.latlng_to_cell(hw["home_lat"], hw["home_lon"], args.coarse_hex_resolution) \
            if hw is not None and pd.notna(hw.get("home_lat")) else None
        work_hex = h3.latlng_to_cell(hw["work_lat"], hw["work_lon"], args.coarse_hex_resolution) \
            if hw is not None and pd.notna(hw.get("work_lat")) else None
        home_zip = hw.get("home_zip") if hw is not None and has_reported_zip else None
        home_zip = None if pd.isna(home_zip) else str(home_zip)
        rec = {"s": r["stop_id"], "m": MODE_CODE.get(r["departure_mode"], 1),
               "dt": 3, "hh": home_hex, "wh": work_hex, "hz": home_zip}

        coords = path_by_key.get(f"{r['device_id']}||{r['stop_id']}")
        if coords:
            rec["ph"] = build_fine_hex_path(coords, args.fine_hex_resolution)
            if segments:
                street_votes = Counter()
                for lon, lat in coords:
                    name = awp.snap_point_to_street(lat, lon, segments, grid, cell_deg, args.max_snap_m)
                    if name:
                        street_votes[name] += 1
                if street_votes:
                    rec["st"] = street_votes.most_common(1)[0][0]
        trip_records.append(rec)

    trips_path = out_dir / "trips.json"
    with open(trips_path, "w") as fh:
        json.dump(trip_records, fh, separators=(",", ":"))
    log.info("Wrote %s (%d trip records, %.1f MB)", trips_path, len(trip_records), trips_path.stat().st_size / 1e6)

    n_riders_by_stop = pd.concat([trips["stop_id"], vehicle["stop_id"]]).value_counts()
    stops_out = [
        {"id": row["stop_id"], "lat": row["lat"], "lon": row["lon"], "routes": row["routes"],
         "address": row.get("address", ""), "n_riders": int(n_riders_by_stop.get(row["stop_id"], 0))}
        for _, row in stops.iterrows() if n_riders_by_stop.get(row["stop_id"], 0) > 0
    ]
    stops_path = out_dir / "stops.json"
    json.dump(stops_out, open(stops_path, "w"), separators=(",", ":"))
    log.info("Wrote %s (%d stops with riders)", stops_path, len(stops_out))

    route_stops: dict[str, list] = {}
    route_riders: Counter = Counter()
    for s in stops_out:
        for route in s["routes"]:
            route_stops.setdefault(route, []).append(s["id"])
            route_riders[route] += s["n_riders"]
    routes_out = [{"route": r, "stop_ids": sids, "n_riders": route_riders[r]} for r, sids in route_stops.items()]
    routes_path = out_dir / "routes.json"
    json.dump(routes_out, open(routes_path, "w"), separators=(",", ":"))
    log.info("Wrote %s (%d routes)", routes_path, len(routes_out))

    categories_out = [{"category": c, "n_trips": n} for c, n in category_counts.most_common()]
    categories_path = out_dir / "categories.json"
    json.dump(categories_out, open(categories_path, "w"), separators=(",", ":"))
    log.info("Wrote %s (%d categories)", categories_path, len(categories_out))

    # Street GEOMETRY, separate from trip counts -- the dashboard aggregates
    # counts per street name client-side (so it stays reactive to filters)
    # and looks up each name's actual line geometry from this file to draw
    # real polylines along the street, rather than the fine hexbin dots
    # (which are only visible zoomed in very close and were easy to mistake
    # for "no path data" at a city-wide view).
    if segments:
        by_name: dict[str, list] = {}
        for seg in segments:
            by_name.setdefault(seg["name"], []).append(
                [[seg["lon1"], seg["lat1"]], [seg["lon2"], seg["lat2"]]]
            )
        street_features = [
            {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": lines},
             "properties": {"name": name}}
            for name, lines in by_name.items()
        ]
        streets_path = out_dir / "streets.geojson"
        json.dump({"type": "FeatureCollection", "features": street_features}, open(streets_path, "w"),
                   separators=(",", ":"))
        log.info("Wrote %s (%d named streets)", streets_path, len(street_features))

    manifest_path = Path(args.out) / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {"pulls": []}
    manifest["pulls"] = [p for p in manifest["pulls"] if p["id"] != args.pull_id]
    manifest["pulls"].append({
        "id": args.pull_id, "n_trips": len(trip_records), "n_stops": len(stops_out), "n_routes": len(routes_out),
        "n_walking": int((trips["departure_mode"] == "walking").sum()) if "departure_mode" in trips.columns else len(trips),
        "has_zips": (out_dir / "zips.geojson").exists(),
    })
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    log.info("Updated %s (%d pull(s) registered)", manifest_path, len(manifest["pulls"]))


if __name__ == "__main__":
    main()