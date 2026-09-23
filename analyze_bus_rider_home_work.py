#!/usr/bin/env python3
"""
analyze_bus_rider_home_work.py
================================
Step 3 of the bus-stop mobility workflow: takes the Azira/Ubermedia device
extract pulled against a bus-stop buffer polygon and answers "where do bus
stop users live and work?" -- per stop, and in aggregate.

THIS SCRIPT DOES NOT QUERY THE PLATFORM. It consumes exports you pull
yourself, using one of the bus_stop_extract_group_*.geojson files (or the
old single merged file) as the query polygon:
  1. --home : a "Common Evening Location" (CEL) extract -- one row per
              device PER OBSERVED NIGHT, giving that device's evening
              location for that night. A device shows up on many different
              dates, usually clustered tightly around one real home (plus
              occasional noise/travel nights elsewhere). See "WHY THIS
              NEEDS AGGREGATION" below.
  2. --work : a "Common Daytime Location" (CDL) extract -- same idea, for
              daytime/work locations.
  3. --pathing : the raw per-ping pathing trail for those same devices,
              used ONLY to compute how long each device actually dwelled
              inside a specific stop's buffer -- the noise filter that
              separates someone waiting for a bus from someone who merely
              drove or walked past the stop once.

WHY THIS NEEDS AGGREGATION (this is the part a naive join gets wrong)
  The CEL/CDL exports are NOT one row per device -- they're one row per
  device PER VISIT/NIGHT. The same device typically has many rows spread
  across many dates, most of them clustered within a block or two of each
  other (the same physical home, with ordinary GPS jitter across nights)
  plus occasionally a handful of rows somewhere completely different
  (travel, a data glitch, or a genuinely secondary location). Naively
  taking "the" home_lat/home_lon per device (e.g. the first row, or an
  unweighted mean of every row) would blend in those outlier nights and
  place "home" somewhere between two real locations, or get pulled off by
  a single stray night.
  Instead, for each device this script clusters its rows by proximity
  (--cluster-tolerance-m, default 150m) and keeps the LARGEST cluster's
  centroid as that device's home (or work) location -- the place it was
  actually seen most consistently -- along with how many distinct dates
  support it (n_nights) as a confidence signal you can filter on.

WORKFLOW
  1. Load bus_stops_clustered.csv (from fetch_and_buffer_bus_stops.py) as
     the stop lookup -- plain lat/lon/radius, no geometry library needed.
  2. For every pathing ping, find every stop within its buffer radius.
  3. Collapse a device's pings-near-a-stop into dwell episodes (gap >
     --max-gap-minutes starts a new episode) and keep only devices with at
     least one episode >= --min-dwell-minutes at that stop -- the core
     noise filter (a single ping with no dwell is almost always a car or
     pedestrian passing by, not a rider).
  4. Join the surviving (device_id, stop_id) pairs to each device's
     aggregated home/work location and summarize per stop.

OUTPUTS
  rider_stop_assignments.csv   -- one row per (device_id, stop_id) that
                                    passed the dwell filter, with dwell
                                    minutes and aggregated home/work coords
  stop_rider_summary.csv       -- per stop: n_riders, n with home/work data
  home_work_map.html           -- Leaflet map: stops sized by rider count,
                                    home locations and work locations as
                                    separate toggleable layers

CAVEATS
  - Column names below match the "expanded cel/cdl detailed report" export
    template (Hashed Ubermedia Id, Common Evening/Daytime Lat/Long, Visit
    Timestamp, ...). If your export differs, run --list-columns FILE first
    and use --column-map to override.
  - A device dwelling near a stop is evidence of *bus stop use*, not proof
    of *boarding a bus*. Treat this as "likely stop users."
  - A device with only 1-2 total CEL/CDL rows produces a "home"/"work"
    location with very low confidence (n_nights=1-2) -- consider filtering
    stop_rider_summary / rider_stop_assignments on a minimum n_nights for
    anything you'd want to publish or map with confidence.

USAGE
  python analyze_bus_rider_home_work.py \\
      --stops bus_stop_output/bus_stops_clustered.csv \\
      --home 10169459_Bus_Test_expanded_cel_cdl_detailed_report_cel.tsv \\
      --work 10169459_Bus_Test_expanded_cel_cdl_detailed_report_cdl.tsv \\
      --pathing 10169459_Bus_Test_pathing_x_context_report/ \\
      --out rider_output/

  # See what columns your actual export uses before mapping them:
  python analyze_bus_rider_home_work.py --list-columns path/to/file.tsv
"""

from __future__ import annotations

import argparse
import glob
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
log = logging.getLogger("analyze_bus_rider_home_work")

DEFAULT_MAX_GAP_MINUTES = 20.0
DEFAULT_MIN_DWELL_MINUTES = 2.0
DEFAULT_CLUSTER_TOLERANCE_M = 150.0

# Candidate column names. The "expanded cel/cdl detailed report" template
# (Ubermedia/Azira) is listed first since that's the real export in use;
# generic fallbacks follow. Override any of these with --column-map.
CANDIDATE_COLUMNS = {
    "device_id": ["Hashed Ubermedia Id", "Hashed Device ID", "device_id", "ifa", "maid", "Device ID"],
    "home_lat": ["Common Evening Lat", "home_lat", "home_latitude", "Home Latitude"],
    "home_lon": ["Common Evening Long", "home_lon", "home_longitude", "Home Longitude"],
    "work_lat": ["Common Daytime Lat", "work_lat", "work_latitude", "Work Latitude"],
    "work_lon": ["Common Daytime Long", "work_lon", "work_longitude", "Work Longitude"],
    "home_postal": ["Common Evening Postal1", "home_postal", "home_zip", "Home Postal", "Home Zip"],
    "work_postal": ["Common Daytime Postal1", "work_postal", "work_zip", "Work Postal", "Work Zip"],
    "visit_date": ["Visit Date", "visit_date"],
    "pathing_lat": ["Lat of Observation Point", "lat", "latitude", "Latitude"],
    "pathing_lon": ["Lon of Observation Point", "lon", "lng", "longitude", "Longitude"],
    "pathing_time": ["Unix Timestamp of Observation Point", "timestamp", "utc_timestamp", "ping_time"],
}


def resolve_column(df: pd.DataFrame, field: str, override: str | None) -> str | None:
    if override:
        if override not in df.columns:
            raise KeyError(f"--column-map gave '{override}' for {field}, but it's not in the file. "
                            f"Available columns: {list(df.columns)}")
        return override
    for cand in CANDIDATE_COLUMNS.get(field, []):
        if cand in df.columns:
            return cand
    return None


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6_371_000.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _sniff_sep(path: str) -> str:
    return "\t" if str(path).lower().endswith((".tsv", ".tsv.gz", ".txt")) else ","


def read_table_any_compression(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Handles the common gotcha in these exports: files are gzip-compressed
    but named .tsv (no .gz suffix), so pandas' extension-based compression
    sniffing doesn't kick in. Tries plain read first, falls back to
    explicit gzip, and vice versa."""
    sep = _sniff_sep(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(2)
        is_gzip_magic = head[:2] == b"\x1f\x8b"
    except Exception:
        is_gzip_magic = False
    compression = "gzip" if is_gzip_magic else None
    return pd.read_csv(path, sep=sep, compression=compression, nrows=nrows, low_memory=False)


def _list_pathing_parts(path: str) -> list:
    """Returns a list of part descriptors: either a plain file path (str)
    for a directory/single-file source, or ("zip", zip_path, entry_name)
    for entries inside a .zip -- so the rest of the pipeline can peek and
    read each part uniformly regardless of source."""
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


def _read_part(part, usecols=None, dtype=None, nrows=None) -> pd.DataFrame:
    """Reads one part (see _list_pathing_parts), sniffing gzip by magic
    bytes rather than trusting the extension, and pulling only usecols
    (with a smaller dtype where given) to keep peak memory down -- these
    bulk pathing exports carry several columns this pipeline never uses."""
    if isinstance(part, tuple):
        _, zip_path, entry_name = part
        sep = _sniff_sep(entry_name)
        with zipfile.ZipFile(zip_path) as zf, zf.open(entry_name) as fh:
            raw = fh.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw), sep=sep, usecols=usecols, dtype=dtype, nrows=nrows, low_memory=False)
    return read_table_any_compression(part, nrows=nrows) if usecols is None else pd.read_csv(
        part, sep=_sniff_sep(part),
        compression="gzip" if open(part, "rb").read(2) == b"\x1f\x8b" else None,
        usecols=usecols, dtype=dtype, nrows=nrows, low_memory=False,
    )


def read_pathing_source(path: str, overrides: dict) -> pd.DataFrame:
    """--pathing may be a single file, a .zip of part files, or a directory
    of already-extracted part files -- bulk pathing exports commonly come
    chunked into many parts rather than one file. Only the 4 columns this
    pipeline needs are pulled from each part (at a smaller dtype for
    lat/lon), and parts are trimmed to those columns BEFORE concatenating
    -- reading every column of every part first (the naive approach) peaks
    at several times the memory this needs and is what was running this
    machine out of RAM on a 20-part, 8GB+ export."""
    parts = _list_pathing_parts(path)
    log.info("  found %d part file(s)", len(parts))

    # Peek the first part's header to resolve real column names once;
    # every part in one export shares the same schema.
    header = _read_part(parts[0], nrows=5)
    dev_c = resolve_column(header, "device_id", overrides.get("device_id"))
    lat_c = resolve_column(header, "pathing_lat", overrides.get("pathing_lat"))
    lon_c = resolve_column(header, "pathing_lon", overrides.get("pathing_lon"))
    time_c = resolve_column(header, "pathing_time", overrides.get("pathing_time"))
    missing = [n for n, c in [("device_id", dev_c), ("lat", lat_c), ("lon", lon_c), ("timestamp", time_c)] if c is None]
    if missing:
        raise KeyError(f"Couldn't auto-detect pathing columns {missing} in {path}. "
                        f"Found: {list(header.columns)}. Use --column-map to override.")

    usecols = [dev_c, lat_c, lon_c, time_c]
    dtype = {lat_c: "float32", lon_c: "float32"}
    frames = []
    for i, part in enumerate(parts):
        frame = _read_part(part, usecols=usecols, dtype=dtype)
        frame = frame.rename(columns={dev_c: "device_id", lat_c: "lat", lon_c: "lon", time_c: "timestamp"})
        frames.append(frame)
        if (i + 1) % 5 == 0 or i + 1 == len(parts):
            log.info("  read part %d/%d (%d rows so far)", i + 1, len(parts), sum(len(f) for f in frames))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_common_location_file(path: str, kind: str, overrides: dict, keep_device_ids: set | None = None) -> pd.DataFrame:
    """Loads a CEL (home) or CDL (work) export and collapses many
    per-visit rows per device down to one dominant location per device
    (see module docstring: 'WHY THIS NEEDS AGGREGATION'). If
    keep_device_ids is given, rows for every other device are dropped
    immediately after reading -- BEFORE the per-device clustering step --
    since a citywide export commonly has 1M+ devices while the qualifying
    rider set is usually a tiny fraction of that; clustering location
    history for devices that were never actually near a stop is wasted
    work and, at this scale, the main reason this step used to be slow.
    Returns columns: device_id, {kind}_lat, {kind}_lon, n_{kind}_nights."""
    log.info("Loading %s extract from %s ...", kind, path)
    df = read_table_any_compression(path)
    device_col = resolve_column(df, "device_id", overrides.get("device_id"))
    lat_col = resolve_column(df, f"{kind}_lat", overrides.get(f"{kind}_lat"))
    lon_col = resolve_column(df, f"{kind}_lon", overrides.get(f"{kind}_lon"))
    date_col = resolve_column(df, "visit_date", overrides.get("visit_date"))
    postal_col = resolve_column(df, f"{kind}_postal", overrides.get(f"{kind}_postal"))
    if device_col is None or lat_col is None or lon_col is None:
        raise KeyError(
            f"Couldn't auto-detect device_id/{kind}_lat/{kind}_lon columns in {path}. "
            f"Found columns: {list(df.columns)}. Use --column-map device_id=... {kind}_lat=... {kind}_lon=..."
        )
    df = df.rename(columns={device_col: "device_id", lat_col: "lat", lon_col: "lon"})
    if date_col:
        df = df.rename(columns={date_col: "visit_date"})
    else:
        df["visit_date"] = np.nan
    if postal_col:
        df = df.rename(columns={postal_col: "postal"})
        log.info("  found a %s postal/zip column ('%s') -- will use it directly instead of "
                  "reverse-geocoding lat/lon.", kind, postal_col)
    else:
        df["postal"] = np.nan
    df = df.dropna(subset=["device_id", "lat", "lon"])
    if keep_device_ids is not None:
        before = df["device_id"].nunique()
        df = df[df["device_id"].isin(keep_device_ids)]
        log.info("  restricted from %d to %d devices (qualifying riders only) before clustering.",
                  before, df["device_id"].nunique())
    log.info("  %d rows across %d devices going into per-device aggregation.", len(df), df["device_id"].nunique())
    return df[["device_id", "lat", "lon", "visit_date", "postal"]]


def dominant_location_per_device(rows: pd.DataFrame, tolerance_m: float, out_lat_col: str, out_lon_col: str,
                                   out_n_col: str, out_postal_col: str | None = None) -> pd.DataFrame:
    """For each device, greedily clusters its (lat, lon) rows by proximity
    and returns the largest cluster's centroid -- see module docstring.
    Also carries through the MODAL postal/zip code among that cluster's
    rows, when the source export has one (e.g. the CEL/CDL "Common Evening/
    Daytime Postal1" field) -- this is a real device-reported zip code,
    not a reverse-geocoding guess, so it's used directly rather than fetched
    from an external boundary service."""
    records = []
    for device_id, g in rows.groupby("device_id"):
        pts = g[["lat", "lon"]].to_numpy()
        dates = g["visit_date"].tolist()
        postals = g["postal"].tolist() if "postal" in g.columns else [None] * len(g)
        clusters: list[dict] = []
        for (lat, lon), date, postal in zip(pts, dates, postals):
            placed = False
            for c in clusters:
                if haversine_m(lat, lon, c["lat_sum"] / c["n"], c["lon_sum"] / c["n"]) <= tolerance_m:
                    c["lat_sum"] += lat
                    c["lon_sum"] += lon
                    c["n"] += 1
                    c["dates"].add(date)
                    c["postals"].append(postal)
                    placed = True
                    break
            if not placed:
                clusters.append({"lat_sum": lat, "lon_sum": lon, "n": 1, "dates": {date}, "postals": [postal]})
        best = max(clusters, key=lambda c: len(c["dates"]))
        record = {
            "device_id": device_id,
            out_lat_col: best["lat_sum"] / best["n"],
            out_lon_col: best["lon_sum"] / best["n"],
            out_n_col: len(best["dates"]),
        }
        if out_postal_col:
            valid_postals = [p for p in best["postals"] if pd.notna(p)]
            record[out_postal_col] = pd.Series(valid_postals).mode().iloc[0] if valid_postals else None
        records.append(record)
    return pd.DataFrame(records)


def assign_pings_to_stops(pathing: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Grid-buckets both pings and stops into coarse lat/lon cells sized to
    the buffer radius, then only checks each ping against stops in its own
    cell + the 8 neighboring cells -- instead of an O(n_stops x n_pings)
    full-dataframe boolean mask repeated once per stop. A ping can land
    within radius of more than one stop (e.g. a shared-corner cluster), so
    this still returns one row per (ping, stop) match, not just the
    nearest stop.

    Grouping pings by cell uses a single vectorized numpy sort, NOT
    pandas' groupby(...).groups -- at real scale (100M+ pings), materializing
    every group's row-index array via pandas' Categorical-based grouping
    machinery is itself what ran out of memory (a single internal int64
    array allocation failed at ~150M pings). A sort + np.unique(return_index)
    finds the same group boundaries in one pass without that per-group
    index-array overhead, and the per-cell loop that follows operates on
    plain numpy slices instead of repeated DataFrame .loc lookups."""
    cell_deg = max(stops["buffer_meters"].max() / 100_000.0, 0.0005)
    stops = stops.reset_index(drop=True)

    device_id_all = pathing["device_id"].to_numpy()
    timestamp_all = pathing["timestamp"].to_numpy()
    lat_all = pathing["lat"].to_numpy()
    lon_all = pathing["lon"].to_numpy()

    # device_id is typically a long hash string (Azira/Ubermedia exports use
    # 40-character hex ids) -- at real scale (100M+ pings) storing that as
    # repeated Python string objects is itself a major memory cost,
    # separate from the grouping fix above. factorize() replaces it with a
    # compact integer code plus one small array of the actual unique
    # strings; codes are mapped back to strings only in the final output.
    device_codes_all, device_uniques = pd.factorize(device_id_all)

    clat_all = np.round(lat_all / cell_deg).astype(np.int64)
    clon_all = np.round(lon_all / cell_deg).astype(np.int64)
    stop_cell_lat = np.round(stops["lat"].to_numpy() / cell_deg).astype(np.int64)
    stop_cell_lon = np.round(stops["lon"].to_numpy() / cell_deg).astype(np.int64)

    stop_by_cell: dict[tuple, list[int]] = {}
    for i, (clat, clon) in enumerate(zip(stop_cell_lat, stop_cell_lon)):
        stop_by_cell.setdefault((int(clat), int(clon)), []).append(i)

    # Combine (clat, clon) into one sortable int64 key. Offset/multiplier
    # are generous relative to the plausible range of clat/clon (lat/lon in
    # degrees divided by a sub-degree cell size), so there's no collision risk.
    OFFSET, MULT = 2_000_000, 5_000_000
    cell_key_all = (clat_all + OFFSET) * MULT + (clon_all + OFFSET)

    order = np.argsort(cell_key_all, kind="stable")
    cell_key_sorted = cell_key_all[order]
    clat_sorted, clon_sorted = clat_all[order], clon_all[order]
    lat_sorted, lon_sorted = lat_all[order], lon_all[order]
    device_code_sorted, timestamp_sorted = device_codes_all[order], timestamp_all[order]
    del order, cell_key_all, clat_all, clon_all, lat_all, lon_all, device_id_all, device_codes_all, timestamp_all

    unique_keys, starts = np.unique(cell_key_sorted, return_index=True)
    ends = np.append(starts[1:], len(cell_key_sorted))
    log.info("  %d pings grouped into %d distinct grid cells", len(cell_key_sorted), len(unique_keys))

    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    device_codes, timestamps, stop_ids_out = [], [], []
    for gi in range(len(unique_keys)):
        start, end = starts[gi], ends[gi]
        clat, clon = int(clat_sorted[start]), int(clon_sorted[start])
        candidate_idx: list[int] = []
        for dx, dy in neighbor_offsets:
            candidate_idx.extend(stop_by_cell.get((clat + dx, clon + dy), []))
        if not candidate_idx:
            continue
        cand = stops.iloc[candidate_idx]
        # Vectorized (n_pings_in_cell x n_candidate_stops) distance matrix.
        lat1 = lat_sorted[start:end][:, None]
        lon1 = lon_sorted[start:end][:, None]
        lat2 = cand["lat"].to_numpy()[None, :]
        lon2 = cand["lon"].to_numpy()[None, :]
        dist = haversine_m(lat1, lon1, lat2, lon2)
        within = dist <= cand["buffer_meters"].to_numpy()[None, :]
        ping_i, stop_j = np.nonzero(within)
        if len(ping_i) == 0:
            continue
        device_codes.append(device_code_sorted[start:end][ping_i])
        timestamps.append(timestamp_sorted[start:end][ping_i])
        stop_ids_out.append(cand["stop_id"].to_numpy()[stop_j])
        if (gi + 1) % 20_000 == 0:
            log.info("  processed %d/%d grid cells", gi + 1, len(unique_keys))

    if not device_codes:
        return pd.DataFrame(columns=["device_id", "timestamp", "stop_id"])
    all_codes = np.concatenate(device_codes)
    return pd.DataFrame({
        "device_id": device_uniques[all_codes],
        "timestamp": np.concatenate(timestamps),
        "stop_id": np.concatenate(stop_ids_out),
    })


def compute_dwell_episodes(assigned: pd.DataFrame, max_gap_minutes: float) -> pd.DataFrame:
    """Fully vectorized -- no Python-level loop at all, over pings OR
    episodes. The previous version ran a Python groupby(["device_id",
    "stop_id"]) (once per distinct pair -- potentially hundreds of
    thousands to millions of pairs at real scale) with a SECOND, nested
    groupby inside that loop. At 34M matched pings that combination was
    what turned this step into multiple hours.

    Instead: sort once (a single C-level multi-key sort, not a Python
    loop), then find episode boundaries via plain array comparisons --  a
    new episode starts wherever device_id or stop_id changes, or the gap
    since the previous ping exceeds max_gap_minutes -- and read off each
    episode's start/end/size directly from the boundary indices. No
    grouping machinery of any kind is invoked."""
    if assigned.empty:
        return pd.DataFrame(columns=["device_id", "stop_id", "dwell_minutes", "n_pings", "episode_start"])

    df = assigned.sort_values(["device_id", "stop_id", "timestamp"], kind="stable")
    device_id = df["device_id"].to_numpy()
    stop_id = df["stop_id"].to_numpy()
    ts = df["timestamp"].to_numpy()
    n = len(df)

    new_episode = np.zeros(n, dtype=bool)
    new_episode[0] = True
    if n > 1:
        changed_group = (device_id[1:] != device_id[:-1]) | (stop_id[1:] != stop_id[:-1])
        gap_minutes = (ts[1:] - ts[:-1]) / np.timedelta64(1, "m")
        new_episode[1:] = changed_group | (gap_minutes > max_gap_minutes)

    starts = np.flatnonzero(new_episode)
    ends = np.append(starts[1:], n)

    episode_start = ts[starts]
    dwell_minutes = (ts[ends - 1] - episode_start) / np.timedelta64(1, "m")
    n_pings = ends - starts

    return pd.DataFrame({
        "device_id": device_id[starts],
        "stop_id": stop_id[starts],
        "dwell_minutes": dwell_minutes,
        "n_pings": n_pings,
        "episode_start": episode_start,
    })


def build_home_work_map(stop_summary: pd.DataFrame, assignments: pd.DataFrame, stops: pd.DataFrame,
                          out_path: Path) -> None:
    center_lat, center_lon = stops["lat"].mean(), stops["lon"].mean()

    stop_points = [
        {"lat": r["lat"], "lon": r["lon"], "stop_id": r["stop_id"], "n_riders": int(r.get("n_riders", 0))}
        for _, r in stop_summary.merge(stops[["stop_id", "lat", "lon"]], on="stop_id", how="left").iterrows()
    ]
    home_points, work_points = [], []
    if "home_lat" in assignments.columns:
        h = assignments.dropna(subset=["home_lat", "home_lon"])
        home_points = h[["home_lat", "home_lon"]].rename(columns={"home_lat": "lat", "home_lon": "lon"}).to_dict("records")
    if "work_lat" in assignments.columns:
        w = assignments.dropna(subset=["work_lat", "work_lon"])
        work_points = w[["work_lat", "work_lon"]].rename(columns={"work_lat": "lat", "work_lon": "lon"}).to_dict("records")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Bus Stop Rider Home/Work</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{{margin:0;height:100%;font-family:-apple-system,sans-serif;background:#0c0d10;}}
  #map{{position:absolute;inset:0;}}
  #legend{{position:absolute;top:12px;right:12px;z-index:1000;background:rgba(12,13,16,.9);
    color:#e8eaf0;padding:12px 16px;border-radius:8px;font-size:13px;line-height:1.9;}}
  #legend label{{display:block;cursor:pointer;}}
  #legend b{{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;vertical-align:middle;}}
</style></head>
<body>
<div id="map"></div>
<div id="legend">
  <label><input type="checkbox" id="toggle-stops" checked><b style="background:#ffb648"></b>Bus stops ({len(stop_points)}, sized by riders)</label>
  <label><input type="checkbox" id="toggle-home" checked><b style="background:#3ec9a7"></b>Rider home locations ({len(home_points)})</label>
  <label><input type="checkbox" id="toggle-work" checked><b style="background:#e85d5d"></b>Rider work locations ({len(work_points)})</label>
</div>
<script>
const STOPS = {json.dumps(stop_points)};
const HOMES = {json.dumps(home_points)};
const WORKS = {json.dumps(work_points)};
const map = L.map('map').setView([{center_lat}, {center_lon}], 12);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20
}}).addTo(map);

const maxRiders = Math.max(1, ...STOPS.map(s => s.n_riders));
const stopLayer = L.layerGroup(STOPS.map(s => L.circleMarker([s.lat, s.lon], {{
  radius: 4 + 10 * Math.sqrt((s.n_riders||0) / maxRiders), color:'#ffb648', weight:1, fillOpacity:0.6
}}).bindTooltip(`${{s.stop_id}} -- ${{s.n_riders}} rider(s)`))).addTo(map);

const homeLayer = L.layerGroup(HOMES.map(p => L.circleMarker([p.lat, p.lon], {{
  radius:3, color:'#3ec9a7', weight:0, fillOpacity:0.5
}}))).addTo(map);

const workLayer = L.layerGroup(WORKS.map(p => L.circleMarker([p.lat, p.lon], {{
  radius:3, color:'#e85d5d', weight:0, fillOpacity:0.5
}}))).addTo(map);

document.getElementById('toggle-stops').addEventListener('change', e => e.target.checked ? map.addLayer(stopLayer) : map.removeLayer(stopLayer));
document.getElementById('toggle-home').addEventListener('change', e => e.target.checked ? map.addLayer(homeLayer) : map.removeLayer(homeLayer));
document.getElementById('toggle-work').addEventListener('change', e => e.target.checked ? map.addLayer(workLayer) : map.removeLayer(workLayer));
</script>
</body></html>"""
    out_path.write_text(html)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stops", help="bus_stops_clustered.csv from fetch_and_buffer_bus_stops.py")
    p.add_argument("--home", help="Common Evening Location (CEL) extract -- one row per device per visit night.")
    p.add_argument("--work", help="Common Daytime Location (CDL) extract -- one row per device per visit day.")
    p.add_argument("--pathing", help="Raw pathing extract: a file, a .zip of parts, or a directory of parts.")
    p.add_argument("--list-columns", metavar="FILE", help="Print FILE's columns and exit.")
    p.add_argument("--column-map", action="append", default=[],
                    help="Override auto-detected column name: field=ActualColumnName. Repeatable. "
                         "Fields: device_id, home_lat, home_lon, work_lat, work_lon, visit_date, "
                         "pathing_lat, pathing_lon, pathing_time.")
    p.add_argument("--cluster-tolerance-m", type=float, default=DEFAULT_CLUSTER_TOLERANCE_M,
                    help="Rows within this many meters, for the same device, are treated as the same home/work "
                         "location when collapsing many per-visit rows down to one location per device.")
    p.add_argument("--max-gap-minutes", type=float, default=DEFAULT_MAX_GAP_MINUTES,
                    help="Ping gap that starts a new dwell episode at a stop.")
    p.add_argument("--min-dwell-minutes", type=float, default=DEFAULT_MIN_DWELL_MINUTES,
                    help="Minimum single-episode dwell time at a stop to count as a likely rider, not a passerby.")
    p.add_argument("--out", default="rider_output/")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.list_columns:
        df = read_table_any_compression(args.list_columns, nrows=5)
        print("Columns in", args.list_columns, ":")
        for c in df.columns:
            print(" -", c)
        return

    if not (args.stops and args.pathing and (args.home or args.work)):
        sys.exit("--stops, --pathing, and at least one of --home/--work are required "
                  "(or pass --list-columns FILE to inspect a file first).")

    overrides = dict(kv.split("=", 1) for kv in args.column_map)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading stop lookup from %s ...", args.stops)
    stops = pd.read_csv(args.stops)

    # Pathing + the dwell filter run FIRST, before touching --home/--work.
    # The qualifying rider set is typically a tiny fraction of the millions
    # of devices in a citywide home/work export -- restricting to that set
    # before the per-device clustering step (not after) is what keeps this
    # fast; clustering location history for every device in the export
    # when only a few thousand ever show up at a stop is wasted work.
    log.info("Loading pathing extract from %s ...", args.pathing)
    pathing = read_pathing_source(args.pathing, overrides)
    pathing["timestamp"] = pd.to_datetime(pathing["timestamp"], unit="s", errors="coerce") if np.issubdtype(
        pathing["timestamp"].dtype, np.number) else pd.to_datetime(pathing["timestamp"], errors="coerce")
    pathing = pathing.dropna(subset=["lat", "lon", "timestamp", "device_id"])
    log.info("Loaded %d pings for %d devices.", len(pathing), pathing["device_id"].nunique())

    log.info("Assigning pings to stop buffers ...")
    assigned = assign_pings_to_stops(pathing, stops)
    log.info("%d pings landed inside a stop buffer.", len(assigned))
    if assigned.empty:
        log.error("No pings fell within any stop buffer -- check that --pathing and --stops cover the same area.")
        sys.exit(1)

    log.info("Computing dwell episodes (max gap %.1f min) ...", args.max_gap_minutes)
    episodes = compute_dwell_episodes(assigned, args.max_gap_minutes)
    qualifying = episodes[episodes["dwell_minutes"] >= args.min_dwell_minutes]
    log.info("%d of %d episodes meet the %.1f-minute dwell threshold (likely riders vs. passersby).",
              len(qualifying), len(episodes), args.min_dwell_minutes)
    best_episode = qualifying.sort_values("dwell_minutes", ascending=False).drop_duplicates(["device_id", "stop_id"])
    rider_ids = set(best_episode["device_id"])
    log.info("%d unique qualifying rider devices -- home/work will only be resolved for these.", len(rider_ids))

    pin_df = None
    if args.home:
        home_rows = load_common_location_file(args.home, "home", overrides, keep_device_ids=rider_ids)
        home_dom = dominant_location_per_device(home_rows, args.cluster_tolerance_m, "home_lat", "home_lon",
                                                  "n_home_nights", out_postal_col="home_zip")
        n_with_zip = home_dom["home_zip"].notna().sum() if "home_zip" in home_dom.columns else 0
        log.info("Resolved a dominant home location for %d of %d rider devices (%d with a reported home zip).",
                  len(home_dom), len(rider_ids), n_with_zip)
        pin_df = home_dom
    if args.work:
        work_rows = load_common_location_file(args.work, "work", overrides, keep_device_ids=rider_ids)
        work_dom = dominant_location_per_device(work_rows, args.cluster_tolerance_m, "work_lat", "work_lon",
                                                  "n_work_nights", out_postal_col="work_zip")
        log.info("Resolved a dominant work location for %d of %d rider devices.", len(work_dom), len(rider_ids))
        pin_df = work_dom if pin_df is None else pin_df.merge(work_dom, on="device_id", how="outer")

    assignments = best_episode.merge(pin_df, on="device_id", how="left") if pin_df is not None else best_episode
    assignments_path = out_dir / "rider_stop_assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    log.info("Wrote %s (%d device-stop assignments)", assignments_path, len(assignments))

    stop_summary = (
        assignments.groupby("stop_id")
        .agg(n_riders=("device_id", "nunique"), avg_dwell_minutes=("dwell_minutes", "mean"))
        .reset_index()
    )
    if "home_lat" in assignments.columns:
        stop_summary["n_riders_with_home"] = assignments.groupby("stop_id")["home_lat"].apply(lambda s: s.notna().sum()).values
    if "work_lat" in assignments.columns:
        stop_summary["n_riders_with_work"] = assignments.groupby("stop_id")["work_lat"].apply(lambda s: s.notna().sum()).values
    summary_path = out_dir / "stop_rider_summary.csv"
    stop_summary.merge(stops[["stop_id", "routes", "address"]], on="stop_id", how="left").to_csv(summary_path, index=False)
    log.info("Wrote %s (%d stops with qualifying riders)", summary_path, len(stop_summary))

    map_path = out_dir / "home_work_map.html"
    build_home_work_map(stop_summary, assignments, stops, map_path)
    log.info("Wrote %s", map_path)


if __name__ == "__main__":
    main()