#!/usr/bin/env python3
"""
trace_departures_and_destinations.py
======================================
Step 4 of the bus-stop mobility workflow: for every qualifying rider-stop
episode in rider_stop_assignments.csv, looks at what the device does
immediately AFTER leaving the stop's buffer, and answers three things:
  1. Did they leave on foot, or in a vehicle? (speed-based, same method
     and default thresholds as classify_travel_mode.py, so results stay
     consistent across the pipeline.)
  2. For the ones who left on foot -- where did they end up? Compared
     against that device's own resolved home/work location (from
     analyze_bus_rider_home_work.py) and, optionally, the nearest
     SafeGraph place, for an "errand" destination.
  3. The literal walked path for each foot departure, as a GeoJSON
     LineString -- this is the input the NEXT pipeline step (snapping
     these paths to the OSM road network) will consume.

WHY "ON FOOT AFTER LEAVING THE STOP" IS THE RIGHT SIGNAL HERE
  This script does not try to distinguish "riding the bus" from "driving a
  car" -- that's not recoverable from GPS speed alone (see
  classify_travel_mode.py's own caveats). Instead it takes the dwell
  filter already applied upstream (rider_stop_assignments.csv) as the bus-
  stop-use signal, and asks a narrower, answerable question: once someone
  who dwelled at a stop starts moving again, are they walking away (likely
  just got off, or gave up waiting and left on foot) or moving at vehicle
  speed (drove off, got picked up, or the dwell was unrelated to boarding
  at all -- e.g. sitting on the bench without ever riding)? Only the
  foot-departures are traced onward to a destination; vehicle departures
  are written to their own file for transparency/QA rather than silently
  dropped.

WORKFLOW
  1. Load rider_stop_assignments.csv and restrict the pathing load to just
     those rider devices (this is the same "filter before the expensive
     part" pattern analyze_bus_rider_home_work.py uses for home/work --
     riders are typically a tiny fraction of a citywide pathing export).
  2. For each rider-stop episode, take that device's pings strictly AFTER
     the episode ends, up to --max-trace-minutes or a gap of
     --max-gap-minutes (whichever comes first -- either one ends the
     trace).
  3. Classify consecutive-ping speed exactly as classify_travel_mode.py
     does (stationary / walking / biking / car / noise, same default
     thresholds). The FIRST non-stationary, non-noise segment's mode is
     the "departure mode" for that episode:
       - "car" or "biking" -> written to vehicle_departures.csv, not
         traced further.
       - "walking" -> traced forward through consecutive walking/
         stationary segments (a brief pause doesn't end the trace) until
         a vehicle-speed segment appears, the trail runs out, or the trace
         window closes. The last point before that is treated as where
         they arrived.
  4. That arrival point is classified as home / work / errand / unresolved:
       - within --home-work-tolerance-m of the device's own resolved home
         location (home_lat/home_lon from --riders) -> "home"
       - else within tolerance of work location -> "work"
       - else, if --safegraph-places is given, the nearest SafeGraph place
         within --max-distance-m -> "errand"
       - else -> "unresolved" (ran out of pings, or nowhere known nearby)

OUTPUTS
  trip_destinations.csv    -- one row per rider-stop episode: departure
                                mode, walk duration/distance, resolved
                                destination type + coordinates + (for
                                errands) SafeGraph place info
  vehicle_departures.csv   -- episodes where the departure was NOT on foot
                                (kept for transparency, not silently
                                dropped)
  walking_paths.geojson    -- one LineString per foot-departure trip, with
                                the actual walked points -- feed this to
                                the road-network-snapping step next

CAVEATS
  - "Departure mode" reflects the first clear movement after the stop
    dwell ends, not a smoothed multi-segment vote (unlike
    classify_travel_mode.py's --smooth-window default) -- a single noisy
    GPS jump right as someone starts walking could occasionally misclassify
    the departure. If you see obviously-wrong departure modes in spot
    checks, this is the first place to loosen.
  - A resolved "home" or "work" destination means the trail ended near
    that device's OWN previously-resolved home/work location -- it does
    not confirm they actually went inside a building, just that they
    ended up in that vicinity.
  - "unresolved" is common and expected: it covers trips where pathing
    coverage simply runs out before a destination is reached (ping
    density drops once someone's out of the original bus-stop extract's
    coverage area), not just genuine data problems.

USAGE
  python trace_departures_and_destinations.py \\
      --riders rider_output/rider_stop_assignments.csv \\
      --pathing 10169459_Bus_Test_pathing_x_context_report.zip \\
      --safegraph-places el_paso_all.geojson \\
      --out trip_output/
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("trace_departures_and_destinations")

# Same defaults as classify_travel_mode.py, kept identical on purpose so
# "walking" means the same thing across both scripts.
DEFAULT_STATIONARY_MAX_MPS = 0.3
DEFAULT_WALK_MAX_MPS = 2.2
DEFAULT_BIKE_MAX_MPS = 7.0
DEFAULT_IMPLAUSIBLE_MPS = 33.0
DEFAULT_MIN_SEGMENT_SECONDS = 10.0

DEFAULT_MAX_TRACE_MINUTES = 120.0
DEFAULT_MAX_GAP_MINUTES = 45.0
DEFAULT_HOME_WORK_TOLERANCE_M = 200.0
DEFAULT_MAX_DISTANCE_M = 100.0
# El Paso - Las Cruces - Ciudad Juarez combined metro area (min_lon, min_lat, max_lon, max_lat).
# Any ping outside this box is dropped before tracing even starts -- see
# trace_one_episode for why this matters: a small number of pathological
# pings (bad GPS/cell-tower fixes, often paired with a near-zero time gap
# to their neighbor, which is exactly the case this pipeline's
# min-segment-seconds check was silently letting slip through unvalidated)
# were producing "walking" traces that jumped to other US states, Mexico,
# and Canada -- physically impossible for an on-foot trip, and the bug
# that prompted this hard geographic clip rather than relying on the
# speed-implausibility check alone to catch it.
DEFAULT_STUDY_BBOX = (-106.95, 31.30, -105.90, 32.60)

CANDIDATE_COLUMNS = {
    "device_id": ["Hashed Device ID", "device_id", "ifa", "maid", "Device ID"],
    "lat": ["Lat of Observation Point", "lat", "latitude", "Latitude"],
    "lon": ["Lon of Observation Point", "lon", "lng", "longitude", "Longitude"],
    "timestamp": ["Unix Timestamp of Observation Point", "timestamp", "utc_timestamp"],
    "sg_lat": ["LATITUDE", "latitude", "lat"],
    "sg_lon": ["LONGITUDE", "longitude", "lon"],
    "placekey": ["PLACEKEY", "placekey", "safegraph_place_id"],
    "location_name": ["LOCATION_NAME", "location_name", "name"],
    "top_category": ["TOP_CATEGORY", "top_category", "sub_category"],
    "parent_placekey": ["PARENT_PLACEKEY", "parent_placekey"],
}


def resolve_column(df: pd.DataFrame, field: str, override: str | None) -> str | None:
    if override:
        if override not in df.columns:
            raise KeyError(f"--column-map gave '{override}' for {field}, not in file. Columns: {list(df.columns)}")
        return override
    for cand in CANDIDATE_COLUMNS.get(field, []):
        if cand in df.columns:
            return cand
    return None


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def classify_speed(speed_mps, stationary_max, walk_max, bike_max, implausible_max) -> str:
    if speed_mps > implausible_max:
        return "noise"
    if speed_mps <= stationary_max:
        return "stationary"
    if speed_mps <= walk_max:
        return "walking"
    if speed_mps <= bike_max:
        return "biking"
    return "car"


def _sniff_sep(path: str) -> str:
    return "\t" if str(path).lower().endswith((".tsv", ".tsv.gz", ".txt")) else ","


def read_table_any_compression(path: str, nrows: int | None = None) -> pd.DataFrame:
    sep = _sniff_sep(path)
    try:
        with open(path, "rb") as fh:
            is_gzip_magic = fh.read(2) == b"\x1f\x8b"
    except Exception:
        is_gzip_magic = False
    return pd.read_csv(path, sep=sep, compression="gzip" if is_gzip_magic else None, nrows=nrows, low_memory=False)


def _list_pathing_parts(path: str) -> list:
    p = Path(path)
    if p.is_dir():
        parts = sorted(str(x) for x in p.rglob("*") if x.is_file()
                        and x.suffix.lower() in (".tsv", ".csv", ".gz", ".txt"))
        if not parts:
            raise FileNotFoundError(f"No .tsv/.csv/.gz part files found under directory {p}")
        return parts
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        return [("zip", str(p), n) for n in names]
    return [str(p)]


def _open_part_stream(part):
    """Returns (binary_file_like_object, sep) for a part, decompressing
    gzip ON THE FLY via gzip.GzipFile wrapping a file handle, rather than
    ever reading a part's full decompressed content into one bytes blob
    (which is what crashed on a single part before any per-part processing
    even started -- a part's decompressed size alone can exceed available
    memory). Both the zip-entry and plain-file cases return a stream
    suitable for pd.read_csv(..., chunksize=...)."""
    if isinstance(part, tuple):
        _, zip_path, entry_name = part
        zf = zipfile.ZipFile(zip_path)  # left open; closed by the caller via the returned stream's close()
        fileobj = zf.open(entry_name)
        if entry_name.lower().endswith(".gz"):
            fileobj = gzip.GzipFile(fileobj=fileobj)
        return fileobj, _sniff_sep(entry_name)
    f = open(part, "rb")
    magic = f.read(2)
    f.seek(0)
    if magic == b"\x1f\x8b":
        f = gzip.GzipFile(fileobj=f)
    return f, _sniff_sep(part)


def _peek_part_header(part) -> pd.DataFrame:
    stream, sep = _open_part_stream(part)
    try:
        return pd.read_csv(stream, sep=sep, nrows=5, low_memory=False)
    finally:
        stream.close()


def build_pathing_by_device(path: str, overrides: dict, keep_device_ids: set, chunksize: int = 500_000) -> dict:
    """Reads a multi-part pathing export and returns {device_id: {"lat":
    np.array, "lon": np.array, "timestamp": np.array of datetime64}}, each
    sorted by timestamp, restricted to keep_device_ids. Three separate
    performance/memory problems in earlier versions of this function are
    addressed together here:
      1. Streaming decompression: each part is streamed through
         gzip.GzipFile directly into pandas' chunked reader (chunksize rows
         at a time) rather than ever decompressing a whole part into one
         bytes blob first -- a single part's full decompressed size can
         exceed available memory on its own, independent of anything done
         across parts.
      2. No cross-part accumulation of full DataFrames: only compact
         per-chunk numpy arrays are kept in memory as parts are read, never
         a growing list of full pandas DataFrames.
      3. No per-chunk, per-device Python loop: device_id is mapped to a
         compact integer code up front, and EVERY chunk's filtered rows are
         appended as whole arrays with no per-device splitting during the
         read loop at all. The actual split into per-device arrays happens
         exactly ONCE, at the end, via a single vectorized lexsort. The
         previous version repeated a small Python-level groupby+iloc loop
         once per chunk across every part -- with ~270k rider devices, that
         adds up to millions of small operations and was the reason a run
         was taking hours rather than minutes; a single sort over the same
         total row count is orders of magnitude faster."""
    parts = _list_pathing_parts(path)
    log.info("  found %d part file(s)", len(parts))
    header = _peek_part_header(parts[0])
    dev_c = resolve_column(header, "device_id", overrides.get("device_id"))
    lat_c = resolve_column(header, "lat", overrides.get("lat"))
    lon_c = resolve_column(header, "lon", overrides.get("lon"))
    time_c = resolve_column(header, "timestamp", overrides.get("timestamp"))
    missing = [n for n, c in [("device_id", dev_c), ("lat", lat_c), ("lon", lon_c), ("timestamp", time_c)] if c is None]
    if missing:
        raise KeyError(f"Couldn't auto-detect pathing columns {missing} in {path}. "
                        f"Found: {list(header.columns)}. Use --column-map to override.")
    usecols = [dev_c, lat_c, lon_c, time_c]
    dtype = {lat_c: "float32", lon_c: "float32"}

    # The previous version split every chunk into per-device groups with a
    # Python-level groupby+iloc loop, run once per chunk across every part
    # -- with up to ~270k distinct rider devices, that's potentially
    # millions of small dict/iloc/to_numpy calls total, which is what was
    # taking hours. Instead: map device_id to a compact integer code up
    # front, accumulate whole-chunk arrays with NO per-device splitting
    # during the read loop at all, and do the split-by-device just ONCE at
    # the very end via a single vectorized sort -- a lexsort over tens of
    # millions of rows typically takes seconds, not the hours a scattered
    # Python loop over the same data takes.
    device_list = sorted(keep_device_ids)
    device_to_code = {d: i for i, d in enumerate(device_list)}

    code_chunks, lat_chunks, lon_chunks, ts_chunks = [], [], [], []
    total_rows = 0
    for i, part in enumerate(parts):
        stream, sep = _open_part_stream(part)
        n_chunks = 0
        try:
            reader = pd.read_csv(stream, sep=sep, usecols=usecols, dtype=dtype,
                                  chunksize=chunksize, low_memory=False)
            for chunk in reader:
                n_chunks += 1
                chunk = chunk.rename(columns={dev_c: "device_id", lat_c: "lat", lon_c: "lon", time_c: "timestamp"})
                chunk["device_id"] = chunk["device_id"].astype(str)
                codes = chunk["device_id"].map(device_to_code)
                mask = codes.notna()
                if not mask.any():
                    continue
                total_rows += int(mask.sum())
                code_chunks.append(codes[mask].to_numpy(dtype=np.int32))
                lat_chunks.append(chunk.loc[mask, "lat"].to_numpy())
                lon_chunks.append(chunk.loc[mask, "lon"].to_numpy())
                ts_chunks.append(chunk.loc[mask, "timestamp"].to_numpy())
        finally:
            stream.close()
        if (i + 1) % 5 == 0 or i + 1 == len(parts):
            log.info("  read part %d/%d (%d chunks, %d rider rows so far)", i + 1, len(parts), n_chunks, total_rows)

    if not code_chunks:
        return {}

    log.info("  sorting %d total rider rows by device (single vectorized pass) ...", total_rows)
    codes_all = np.concatenate(code_chunks); del code_chunks
    lat_all = np.concatenate(lat_chunks); del lat_chunks
    lon_all = np.concatenate(lon_chunks); del lon_chunks
    ts_all = np.concatenate(ts_chunks); del ts_chunks

    order = np.lexsort((ts_all, codes_all))  # primary: device code, secondary: timestamp
    codes_sorted = codes_all[order]
    lat_sorted, lon_sorted, ts_sorted = lat_all[order], lon_all[order], ts_all[order]
    del codes_all, lat_all, lon_all, ts_all, order

    ts_sorted = pd.to_datetime(ts_sorted, unit="s", errors="coerce") if np.issubdtype(ts_sorted.dtype, np.number) \
        else pd.to_datetime(ts_sorted, errors="coerce").to_numpy()

    unique_codes, starts = np.unique(codes_sorted, return_index=True)
    ends = np.append(starts[1:], len(codes_sorted))
    log.info("  building per-device arrays for %d devices ...", len(unique_codes))
    result = {}
    for code, start, end in zip(unique_codes, starts, ends):
        result[device_list[code]] = {
            "lat": lat_sorted[start:end], "lon": lon_sorted[start:end], "timestamp": ts_sorted[start:end],
        }
    return result


def load_safegraph_places(path: str, overrides: dict) -> pd.DataFrame | None:
    if not path:
        return None
    if str(path).lower().endswith((".geojson", ".json")):
        data = json.load(open(path))
        places = pd.DataFrame([feat["properties"] for feat in data["features"]])
    else:
        places = read_table_any_compression(path)
    lat_c = resolve_column(places, "sg_lat", overrides.get("sg_lat"))
    lon_c = resolve_column(places, "sg_lon", overrides.get("sg_lon"))
    pk_c = resolve_column(places, "placekey", overrides.get("placekey"))
    name_c = resolve_column(places, "location_name", overrides.get("location_name"))
    cat_c = resolve_column(places, "top_category", overrides.get("top_category"))
    parent_c = resolve_column(places, "parent_placekey", overrides.get("parent_placekey"))
    if lat_c is None or lon_c is None or pk_c is None:
        raise KeyError(f"Couldn't auto-detect lat/lon/placekey columns in {path}. Found: {list(places.columns)}")
    places = places.rename(columns={lat_c: "lat", lon_c: "lon", pk_c: "placekey"})
    if name_c:
        places = places.rename(columns={name_c: "location_name"})
    if cat_c:
        places = places.rename(columns={cat_c: "top_category"})
    places = places.dropna(subset=["lat", "lon", "placekey"]).reset_index(drop=True)

    # Roll sub-places (ATMs, kiosks, counters inside a larger venue -- e.g.
    # a Coinstar or a Bitcoin ATM physically inside a gas station) up to
    # their parent venue's name, using SafeGraph's own PARENT_PLACEKEY
    # field. Matching still uses each place's OWN lat/lon (the kiosk's
    # exact location is a fine proxy for "arrived at this venue"), but the
    # NAME reported for aggregation is the parent's -- otherwise several
    # genuinely co-located destinations fragment into separate small
    # buckets (e.g. "Circle K", "Coinstar", "CO-OP Network ATM" as three
    # separate entries in top_destinations.csv when they're one store).
    if parent_c and "location_name" in places.columns:
        placekey_to_name = dict(zip(places["placekey"], places["location_name"]))
        parent_keys = places[parent_c]
        rolled_up_name = parent_keys.map(placekey_to_name)
        places["location_name"] = rolled_up_name.fillna(places["location_name"])
        n_rolled = int(rolled_up_name.notna().sum())
        if n_rolled:
            log.info("  rolled up %d sub-places (kiosks/ATMs/counters) to their parent venue's name.", n_rolled)
    return places


def nearest_place(lat: float, lon: float, places: pd.DataFrame, max_distance_m: float):
    if places is None or places.empty:
        return None
    dist = haversine_m(lat, lon, places["lat"].to_numpy(), places["lon"].to_numpy())
    best = np.argmin(dist)
    if dist[best] <= max_distance_m:
        row = places.iloc[best]
        return {"placekey": row["placekey"], "location_name": row.get("location_name"),
                "top_category": row.get("top_category"), "distance_m": float(dist[best])}
    return None


def trace_one_episode(device_pings: dict, episode_end, args) -> dict:
    """device_pings: {"lat": np.array, "lon": np.array, "timestamp": np.array
    of datetime64}, already sorted by timestamp. Returns a dict describing
    the departure mode and, for foot departures, the traced path + endpoint."""
    ts = device_pings["timestamp"]
    after_mask = ts > np.datetime64(episode_end)
    if after_mask.sum() < 2:
        return {"departure_mode": "unresolved", "reason": "no_pings_after_episode"}

    trace_end = episode_end + pd.Timedelta(minutes=args.max_trace_minutes)
    window_mask = after_mask & (ts <= np.datetime64(trace_end))
    lat = device_pings["lat"][window_mask]
    lon = device_pings["lon"][window_mask]
    t = ts[window_mask]

    # Hard geographic clip: drop any ping outside the study area entirely,
    # BEFORE computing any distance/speed from it. This is a deliberate
    # blunt instrument -- a bad ping that never enters the arrays at all
    # can't corrupt a distance/speed calculation on either side of it,
    # which is not something the speed-implausibility check alone
    # guaranteed (see DEFAULT_STUDY_BBOX comment above).
    min_lon, min_lat, max_lon, max_lat = args.study_bbox
    in_bbox = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
    n_dropped = int((~in_bbox).sum())
    lat, lon, t = lat[in_bbox], lon[in_bbox], t[in_bbox]
    if len(t) < 2:
        reason = "no_pings_in_trace_window" if n_dropped == 0 else "only_out_of_area_pings_in_trace_window"
        return {"departure_mode": "unresolved", "reason": reason}

    dist_m = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    dt_s = (t[1:] - t[:-1]) / np.timedelta64(1, "s")

    # Phase 1: classify every valid segment first, stopping at the first
    # gap that exceeds max_gap_minutes. This is done as its own pass
    # (rather than deciding the departure mode segment-by-segment as we
    # went, the previous approach) specifically to allow the correction
    # below: a car pulling away from a curb accelerates THROUGH the
    # walking/biking speed range for its first few seconds before reaching
    # cruising speed, so "first non-stationary segment is bike-speed" is
    # frequently just a car's initial acceleration, not an actual cyclist.
    classified = []  # list of (index_i, mode)
    for i in range(len(dist_m)):
        if dt_s[i] / 60.0 > args.max_gap_minutes:
            break
        if dt_s[i] < args.min_segment_seconds:
            continue
        speed = dist_m[i] / dt_s[i]
        mode = classify_speed(speed, args.stationary_max_mps, args.walk_max_mps, args.bike_max_mps, args.implausible_mps)
        if mode == "noise":
            continue
        classified.append((i, mode))

    moving = [(i, m) for i, m in classified if m != "stationary"]
    if not moving:
        return {"departure_mode": "unresolved", "reason": "all_stationary_or_noise"}

    departure_mode = moving[0][1]
    if departure_mode == "biking" and len(moving) > 1 and moving[1][1] == "car":
        # The first reading was bike-speed but the very next moving reading
        # is car-speed -- almost certainly a vehicle accelerating away from
        # the stop, not someone on a bike who then switched to a car mid-block.
        departure_mode = "car"

    if departure_mode not in ("walking", "car", "biking"):
        return {"departure_mode": departure_mode}

    # Phase 2: build the accumulated path from the classified segments.
    # For walking, this runs until a vehicle-speed segment appears (someone
    # walking away from the stop, unless they get in a car mid-block). For
    # car/biking, this previously captured NOTHING at all -- meaning the
    # dashboard could never show which street a driver/cyclist actually
    # pulled onto, only that they existed. Capturing a short trace here (
    # capped at --vehicle-trace-minutes, since a car covers far more ground
    # per minute than a walker and a full city-crossing polyline isn't
    # useful -- just the local departure street is) fixes that.
    path_lat, path_lon, path_t = [lat[0]], [lon[0]], [t[0]]
    if departure_mode == "walking":
        for i, mode in classified:
            if mode in ("car", "biking"):
                break
            path_lat.append(lat[i + 1]); path_lon.append(lon[i + 1]); path_t.append(t[i + 1])
    else:
        cutoff = pd.Timestamp(t[0]) + pd.Timedelta(minutes=args.vehicle_trace_minutes)
        for i, mode in classified:
            if pd.Timestamp(t[i + 1]) > cutoff:
                break
            path_lat.append(lat[i + 1]); path_lon.append(lon[i + 1]); path_t.append(t[i + 1])

    path_lat = np.array(path_lat); path_lon = np.array(path_lon); path_t = np.array(path_t)
    walk_distance_m = float(haversine_m(path_lat[:-1], path_lon[:-1], path_lat[1:], path_lon[1:]).sum()) \
        if len(path_lat) > 1 else 0.0
    walk_duration_min = (pd.Timestamp(path_t[-1]) - pd.Timestamp(path_t[0])).total_seconds() / 60.0

    if departure_mode != "walking":
        # Car/biking: just the short departure path for street-snapping --
        # no destination resolution (that's out of scope for a vehicle trip,
        # which can go anywhere in the city; only the local departure street
        # is meaningful here).
        return {
            "departure_mode": departure_mode,
            "walk_distance_m": walk_distance_m, "walk_duration_min": walk_duration_min,
            "n_walk_pings": len(path_lat),
            "path": list(zip(path_lon.tolist(), path_lat.tolist())),
        }

    # Phase 3: find the ARRIVAL DWELL -- the real methodological fix. Rather
    # than declaring "wherever the trail happened to stop" the destination
    # (which is wrong whenever the trail simply ran out mid-walk), scan the
    # walked path for the LAST run of consecutive pings that stay within
    # arrival_radius_m of each other for at least min_destination_dwell_
    # minutes: i.e. actual evidence the walker stopped and stayed somewhere.
    # The centroid of that dwell cluster is the destination, and its
    # existence is what dwell_confirmed reports. When no such dwell exists,
    # the endpoint is reported but flagged dwell_confirmed=False, and the
    # arrival point falls back to the last ping (old behavior) so nothing
    # that resolved before still silently disappears -- but you can now
    # filter to dwell_confirmed rows for the trustworthy subset.
    arrival_lat, arrival_lon = float(path_lat[-1]), float(path_lon[-1])
    dwell_confirmed = False
    dwell_minutes_at_dest = 0.0
    if args.min_destination_dwell_minutes > 0 and len(path_t) >= 2:
        # Walk backwards from the end: grow a cluster of trailing pings as
        # long as they stay within arrival_radius_m of the running centroid.
        # Stop when a ping falls outside -- that's the boundary of the final
        # stationary cluster. If that cluster spans >= the dwell threshold
        # in time, it's a confirmed arrival.
        cx, cy = float(path_lat[-1]), float(path_lon[-1])
        sum_lat, sum_lon, cnt = cx, cy, 1
        j = len(path_lat) - 2
        while j >= 0:
            if haversine_m(path_lat[j], path_lon[j], sum_lat / cnt, sum_lon / cnt) <= args.arrival_radius_m:
                sum_lat += path_lat[j]; sum_lon += path_lon[j]; cnt += 1
                j -= 1
            else:
                break
        cluster_start_idx = j + 1
        dwell_minutes_at_dest = (pd.Timestamp(path_t[-1]) - pd.Timestamp(path_t[cluster_start_idx])).total_seconds() / 60.0
        if dwell_minutes_at_dest >= args.min_destination_dwell_minutes and cnt >= 2:
            dwell_confirmed = True
            arrival_lat = float(sum_lat / cnt)
            arrival_lon = float(sum_lon / cnt)

    result = {
        "departure_mode": "walking",
        "end_lat": arrival_lat, "end_lon": arrival_lon,
        "walk_distance_m": walk_distance_m, "walk_duration_min": walk_duration_min,
        "n_walk_pings": len(path_lat),
        "dwell_confirmed": dwell_confirmed if args.min_destination_dwell_minutes > 0 else None,
        "dwell_minutes_at_dest": dwell_minutes_at_dest,
        "path": list(zip(path_lon.tolist(), path_lat.tolist())),  # GeoJSON order: (lon, lat)
    }
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--riders", required=True, help="rider_stop_assignments.csv from analyze_bus_rider_home_work.py")
    p.add_argument("--pathing", required=True, help="Raw pathing extract: a file, a .zip of parts, or a directory.")
    p.add_argument("--safegraph-places", default=None, help="SafeGraph Places GeoJSON or CSV (optional, for errand destinations).")
    p.add_argument("--column-map", action="append", default=[], help="field=ActualColumnName, repeatable.")
    p.add_argument("--stationary-max-mps", type=float, default=DEFAULT_STATIONARY_MAX_MPS)
    p.add_argument("--walk-max-mps", type=float, default=DEFAULT_WALK_MAX_MPS)
    p.add_argument("--bike-max-mps", type=float, default=DEFAULT_BIKE_MAX_MPS)
    p.add_argument("--implausible-mps", type=float, default=DEFAULT_IMPLAUSIBLE_MPS)
    p.add_argument("--min-segment-seconds", type=float, default=DEFAULT_MIN_SEGMENT_SECONDS)
    p.add_argument("--max-trace-minutes", type=float, default=DEFAULT_MAX_TRACE_MINUTES,
                    help="How far forward (in time) to look for a destination after a stop episode ends.")
    p.add_argument("--max-gap-minutes", type=float, default=DEFAULT_MAX_GAP_MINUTES,
                    help="A ping gap larger than this ends the trace (trail went cold).")
    p.add_argument("--home-work-tolerance-m", type=float, default=DEFAULT_HOME_WORK_TOLERANCE_M,
                    help="How close the arrival point must be to the device's own home/work location to count.")
    p.add_argument("--min-destination-dwell-minutes", type=float, default=2.0,
                    help="Core arrival logic (default 2 min, set to 0 to disable). The destination is anchored on "
                         "the LAST cluster of trailing pings that stay within --arrival-radius-m of each other for "
                         "at least this long -- i.e. where the walker actually stopped and stayed, not just where "
                         "the trail happened to end. dwell_confirmed=True means such a cluster was found; when it's "
                         "False the last ping is still used as a best-effort fallback so nothing silently drops.")
    p.add_argument("--arrival-radius-m", type=float, default=60.0,
                    help="Radius defining 'stayed in one place' when detecting the arrival dwell cluster.")
    p.add_argument("--vehicle-trace-minutes", type=float, default=5.0,
                    help="How long after leaving the stop to keep tracing a car/biking departure, for street-"
                         "snapping only -- no destination is resolved for vehicle trips, just the local street "
                         "they pulled onto. Short on purpose: a vehicle covers far more ground per minute than a "
                         "walker, so a long window would trace most of a city-wide drive, not just the departure.")
    p.add_argument("--max-distance-m", type=float, default=DEFAULT_MAX_DISTANCE_M,
                    help="How close the arrival point must be to a SafeGraph place to count as an errand.")
    p.add_argument("--study-bbox", default=",".join(str(x) for x in DEFAULT_STUDY_BBOX),
                    help="min_lon,min_lat,max_lon,max_lat -- pings outside this box are dropped before tracing. "
                         "Default covers the El Paso-Las Cruces-Ciudad Juarez metro area.")
    p.add_argument("--pathing-chunksize", type=int, default=500_000,
                    help="Rows read at a time from each pathing part file. Lower this (e.g. 100000) if the "
                         "machine still runs out of memory on a single very large part.")
    p.add_argument("--out", default="trip_output/")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.study_bbox = tuple(float(x) for x in args.study_bbox.split(","))
    if len(args.study_bbox) != 4:
        sys.exit("--study-bbox must be min_lon,min_lat,max_lon,max_lat")
    overrides = dict(kv.split("=", 1) for kv in args.column_map)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Study area bbox: %s (pings outside this are dropped before tracing)", args.study_bbox)

    log.info("Loading riders from %s ...", args.riders)
    riders = pd.read_csv(args.riders)
    riders["episode_start"] = pd.to_datetime(riders["episode_start"])
    riders["episode_end"] = riders["episode_start"] + pd.to_timedelta(riders["dwell_minutes"], unit="m")
    riders["device_id"] = riders["device_id"].astype(str)
    rider_ids = set(riders["device_id"])
    log.info("%d rider-stop episodes across %d devices.", len(riders), len(rider_ids))

    log.info("Loading pathing from %s (restricted to rider devices) ...", args.pathing)
    pathing_by_device = build_pathing_by_device(args.pathing, overrides, rider_ids, chunksize=args.pathing_chunksize)
    log.info("%d pings loaded for %d rider devices.",
              sum(len(g["lat"]) for g in pathing_by_device.values()), len(pathing_by_device))

    places = None
    if args.safegraph_places:
        log.info("Loading SafeGraph places from %s ...", args.safegraph_places)
        places = load_safegraph_places(args.safegraph_places, overrides)
        log.info("%d places loaded.", len(places))

    trip_rows, vehicle_rows, path_features, vehicle_path_features, unresolved_endpoint_features = [], [], [], [], []
    for _, r in riders.iterrows():
        device_pings = pathing_by_device.get(r["device_id"])
        base = {"device_id": r["device_id"], "stop_id": r["stop_id"],
                "episode_start": r["episode_start"], "episode_end": r["episode_end"]}
        if device_pings is None or len(device_pings["lat"]) < 2:
            trip_rows.append({**base, "departure_mode": "unresolved", "destination_type": "unresolved",
                               "reason": "no_pathing_for_device"})
            continue

        result = trace_one_episode(device_pings, r["episode_end"], args)

        if result["departure_mode"] in ("car", "biking"):
            vehicle_rows.append({**base, "departure_mode": result["departure_mode"]})
            if result.get("path") and len(result["path"]) >= 2:
                vehicle_path_features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in result["path"]]},
                    "properties": {"device_id": r["device_id"], "stop_id": r["stop_id"],
                                    "departure_mode": result["departure_mode"]},
                })
            continue
        if result["departure_mode"] == "unresolved":
            trip_rows.append({**base, "departure_mode": "unresolved", "destination_type": "unresolved",
                               "reason": result.get("reason", "")})
            continue

        # Foot departure -- resolve destination type as before. If dwell
        # confirmation was requested, dwell_confirmed is reported as a
        # SEPARATE confidence column rather than gating the match itself --
        # an early test showed hard-blocking on this wiped out nearly every
        # resolution, including obviously-correct ones, whenever trailing
        # ping density near the destination was thin (which is common --
        # devices often ping less frequently once stationary). Reporting it
        # alongside the result lets you filter to high-confidence matches
        # yourself without losing the rest of the picture.
        end_lat, end_lon = result["end_lat"], result["end_lon"]
        dwell_confirmed = result.get("dwell_confirmed")
        destination_type, dest_info = "unresolved", {}
        home_lat, home_lon = r.get("home_lat"), r.get("home_lon")
        work_lat, work_lon = r.get("work_lat"), r.get("work_lon")
        # Diagnostic distances -- computed unconditionally, not just
        # pass/fail against the current tolerances, so you can tell WHY an
        # endpoint didn't resolve: "missed by 40m" (tolerance is too tight)
        # reads very differently from "nearest place is 2km away" (nothing
        # to match here, no tolerance would fix it).
        nearest_home_m = float(haversine_m(end_lat, end_lon, home_lat, home_lon)) if pd.notna(home_lat) else None
        nearest_work_m = float(haversine_m(end_lat, end_lon, work_lat, work_lon)) if pd.notna(work_lat) else None
        nearest_place_m, nearest_place_name = None, None
        if places is not None and not places.empty:
            dist_all = haversine_m(end_lat, end_lon, places["lat"].to_numpy(), places["lon"].to_numpy())
            nearest_idx = np.argmin(dist_all)
            nearest_place_m = float(dist_all[nearest_idx])
            nearest_place_name = places.iloc[nearest_idx].get("location_name")

        if pd.notna(home_lat) and haversine_m(end_lat, end_lon, home_lat, home_lon) <= args.home_work_tolerance_m:
            destination_type = "home"
        elif pd.notna(work_lat) and haversine_m(end_lat, end_lon, work_lat, work_lon) <= args.home_work_tolerance_m:
            destination_type = "work"
        elif places is not None:
            match = nearest_place(end_lat, end_lon, places, args.max_distance_m)
            if match:
                destination_type = "errand"
                dest_info = match

        trip_rows.append({
            **base, "departure_mode": "walking", "destination_type": destination_type,
            "destination_lat": end_lat, "destination_lon": end_lon,
            "walk_distance_m": result["walk_distance_m"], "walk_duration_min": result["walk_duration_min"],
            "n_walk_pings": result["n_walk_pings"], "dwell_confirmed": dwell_confirmed,
            "dwell_minutes_at_dest": result.get("dwell_minutes_at_dest"),
            "destination_placekey": dest_info.get("placekey"), "destination_name": dest_info.get("location_name"),
            "destination_category": dest_info.get("top_category"),
            "nearest_home_m": nearest_home_m, "nearest_work_m": nearest_work_m,
            "nearest_place_m": nearest_place_m, "nearest_place_name": nearest_place_name,
        })
        path_features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in result["path"]]},
            "properties": {"device_id": r["device_id"], "stop_id": r["stop_id"],
                            "destination_type": destination_type, "destination_name": dest_info.get("location_name")},
        })
        if destination_type == "unresolved":
            unresolved_endpoint_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(end_lon), float(end_lat)]},
                "properties": {
                    "device_id": r["device_id"], "stop_id": r["stop_id"],
                    "walk_duration_min": result["walk_duration_min"], "walk_distance_m": result["walk_distance_m"],
                    "n_walk_pings": result["n_walk_pings"],
                    "nearest_home_m": nearest_home_m, "nearest_work_m": nearest_work_m,
                    "nearest_place_m": nearest_place_m, "nearest_place_name": nearest_place_name,
                },
            })

    trips_df = pd.DataFrame(trip_rows)
    trips_path = out_dir / "trip_destinations.csv"
    trips_df.to_csv(trips_path, index=False)
    log.info("Wrote %s (%d trips)", trips_path, len(trips_df))
    if not trips_df.empty:
        log.info("Departure mode breakdown: %s", trips_df["departure_mode"].value_counts().to_dict())
        foot = trips_df[trips_df["departure_mode"] == "walking"]
        if not foot.empty:
            log.info("Destination breakdown (foot departures only): %s", foot["destination_type"].value_counts().to_dict())
            unresolved_foot = foot[foot["destination_type"] == "unresolved"]
            if not unresolved_foot.empty:
                near_home = (unresolved_foot["nearest_home_m"] <= 2 * args.home_work_tolerance_m).sum()
                near_work = (unresolved_foot["nearest_work_m"] <= 2 * args.home_work_tolerance_m).sum()
                near_place = (unresolved_foot["nearest_place_m"] <= 2 * args.max_distance_m).sum()
                n = len(unresolved_foot)
                log.info(
                    "Of %d unresolved walking destinations: %d (%.0f%%) are within 2x the home/work tolerance of "
                    "home, %d (%.0f%%) within 2x of work, %d (%.0f%%) within 2x the SafeGraph match radius of "
                    "SOME place -- these are the ones widening a threshold could plausibly recover. The rest are "
                    "genuinely far from anything known (see unresolved_endpoints.geojson).",
                    n, near_home, 100 * near_home / n, near_work, 100 * near_work / n, near_place, 100 * near_place / n,
                )

    vehicle_df = pd.DataFrame(vehicle_rows)
    vehicle_path = out_dir / "vehicle_departures.csv"
    vehicle_df.to_csv(vehicle_path, index=False)
    log.info("Wrote %s (%d non-foot departures, kept for QA)", vehicle_path, len(vehicle_df))

    vehicle_paths_path = out_dir / "vehicle_paths.geojson"
    with open(vehicle_paths_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": vehicle_path_features}, fh)
    log.info("Wrote %s (%d car/biking departure paths, for street-snapping only -- no destination is resolved "
              "for these)", vehicle_paths_path, len(vehicle_path_features))

    paths_path = out_dir / "walking_paths.geojson"
    with open(paths_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": path_features}, fh)
    log.info("Wrote %s (%d walking paths)", paths_path, len(path_features))

    unresolved_path = out_dir / "unresolved_endpoints.geojson"
    with open(unresolved_path, "w") as fh:
        json.dump({"type": "FeatureCollection", "features": unresolved_endpoint_features}, fh)
    log.info("Wrote %s (%d unresolved endpoints) -- map this to see WHY they didn't resolve: "
              "nearby-but-just-outside-tolerance (nearest_home_m/nearest_work_m/nearest_place_m close to your "
              "thresholds) vs. genuinely nowhere near anything known (a real coverage gap, not a fixable threshold).",
              unresolved_path, len(unresolved_endpoint_features))


if __name__ == "__main__":
    main()