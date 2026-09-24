#!/usr/bin/env python3
"""
merge_pulls_for_dashboard.py
=============================
Merges every pull already sitting in dashboard/data/<pull_id>/ into one
combined dataset at dashboard/data/combined/, so the dashboard can show
the aggregate of ALL pulls by default instead of one pull at a time.

THIS IS DELIBERATELY CHEAP. It does NOT re-run any of the expensive
pipeline steps (no OSM fetch, no re-tracing pings, no re-resolving
home/work) -- it just reads each pull's already-computed trips.json /
stops.json / routes.json / categories.json / streets.geojson and merges
them. Run this after EVERY new pull's build_dashboard_data.py finishes;
it takes seconds, not minutes, because it's just concatenating and
re-summing JSON that's already been computed.

USAGE
  # After running build_dashboard_data.py for a new pull:
  python merge_pulls_for_dashboard.py --data-dir dashboard/data/

WHAT GETS MERGED
  stops.json      -- the stop network itself is identical across pulls
                      (same city, same GIS layer), so this is read from
                      whichever pull has it; n_riders is recomputed from
                      the COMBINED trips (not summed per-pull) so it's
                      always correct even if pulls overlap in coverage.
  routes.json     -- same idea: recomputed from combined trips + stops.
  categories.json -- summed by category name across pulls.
  streets.geojson -- unioned by street name across pulls (deduped --
                      street geometry from OSM is the same regardless of
                      which pull snapped to it).

  trips.json is NOT pre-merged into one file -- a merged file across even
  2 pulls already exceeded GitHub's 100MB per-file hard limit (125MB from
  just 2 of 8 planned pulls), and would only get worse. Instead, the
  combined manifest entry lists source_pulls (each pull's own id), and
  the dashboard fetches and concatenates each pull's OWN trips.json (each
  one safely under the limit on its own) at load time -- see loadPull()
  in dashboard/index.html, which special-cases is_combined entries to do
  exactly this. Nothing about trips.json needs to change as more pulls
  are added; only stops/routes/categories/streets get re-merged here.

The dashboard needs minimal code for this (already shipped) -- see
dashboard/index.html's loadManifest()/loadPull() for where the combined
view's default selection and multi-pull trip fetch are handled.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("merge_pulls")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="dashboard/data/", help="The dashboard's data/ folder (contains manifest.json).")
    p.add_argument("--combined-id", default="combined", help="Folder name for the merged output (default: combined).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        log.error("No manifest.json at %s -- run build_dashboard_data.py for at least one pull first.", manifest_path)
        return 1
    manifest = json.loads(manifest_path.read_text())

    pull_ids = [p["id"] for p in manifest["pulls"] if p["id"] != args.combined_id]
    if not pull_ids:
        log.error("No pulls found in manifest.json besides '%s' itself -- nothing to merge.", args.combined_id)
        return 1
    log.info("Merging %d pull(s): %s", len(pull_ids), ", ".join(pull_ids))

    all_trips = []  # only used here to recompute stop/route rider counts and totals -- NOT written to disk
    category_counts: Counter = Counter()
    street_features_by_name: dict[str, dict] = {}
    stops_by_id: dict[str, dict] = {}
    stop_routes: dict[str, set] = {}

    for pull_id in pull_ids:
        pull_dir = data_dir / pull_id
        trips = json.loads((pull_dir / "trips.json").read_text())
        all_trips.extend(trips)
        log.info("  %s: %d trips", pull_id, len(trips))

        cats = json.loads((pull_dir / "categories.json").read_text())
        for c in cats:
            category_counts[c["category"]] += c["n_trips"]

        stops = json.loads((pull_dir / "stops.json").read_text())
        for s in stops:
            stops_by_id[s["id"]] = {"id": s["id"], "lat": s["lat"], "lon": s["lon"],
                                     "routes": s["routes"], "address": s.get("address", "")}
            stop_routes.setdefault(s["id"], set()).update(s["routes"])

        streets_path = pull_dir / "streets.geojson"
        if streets_path.exists():
            streets = json.loads(streets_path.read_text())
            for f in streets["features"]:
                name = f["properties"]["name"]
                street_features_by_name.setdefault(name, f)

    log.info("Combined: %d total trips across %d pulls", len(all_trips), len(pull_ids))

    riders_by_stop: Counter = Counter()
    for t in all_trips:
        riders_by_stop[t["s"]] += 1

    stops_out = [
        {**stop, "n_riders": riders_by_stop.get(stop_id, 0)}
        for stop_id, stop in stops_by_id.items() if riders_by_stop.get(stop_id, 0) > 0
    ]

    route_riders: Counter = Counter()
    route_stops: dict[str, list] = {}
    for s in stops_out:
        for route in s["routes"]:
            route_stops.setdefault(route, []).append(s["id"])
            route_riders[route] += s["n_riders"]
    routes_out = [{"route": r, "stop_ids": sids, "n_riders": route_riders[r]} for r, sids in route_stops.items()]

    categories_out = [{"category": c, "n_trips": n} for c, n in category_counts.most_common()]

    out_dir = data_dir / args.combined_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # trips.json is deliberately NOT written here -- see module docstring.
    # The dashboard fetches each source pull's own trips.json directly and
    # concatenates them client-side for the combined view.

    json.dump(stops_out, open(out_dir / "stops.json", "w"), separators=(",", ":"))
    log.info("Wrote %s (%d stops)", out_dir / "stops.json", len(stops_out))

    json.dump(routes_out, open(out_dir / "routes.json", "w"), separators=(",", ":"))
    log.info("Wrote %s (%d routes)", out_dir / "routes.json", len(routes_out))

    json.dump(categories_out, open(out_dir / "categories.json", "w"), separators=(",", ":"))
    log.info("Wrote %s (%d categories)", out_dir / "categories.json", len(categories_out))

    if street_features_by_name:
        streets_out_path = out_dir / "streets.geojson"
        json.dump({"type": "FeatureCollection", "features": list(street_features_by_name.values())},
                   open(streets_out_path, "w"), separators=(",", ":"))
        log.info("Wrote %s (%d named streets)", streets_out_path, len(street_features_by_name))

    manifest["pulls"] = [p for p in manifest["pulls"] if p["id"] != args.combined_id]
    manifest["pulls"].append({
        "id": args.combined_id, "n_trips": len(all_trips), "n_stops": len(stops_out), "n_routes": len(routes_out),
        "n_walking": sum(1 for t in all_trips if t.get("m") == 0),
        "is_combined": True, "source_pulls": pull_ids,
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Updated %s -- combined view now covers: %s", manifest_path, ", ".join(pull_ids))
    log.info("Done. Re-run this script after every new pull's build_dashboard_data.py finishes to keep it current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())