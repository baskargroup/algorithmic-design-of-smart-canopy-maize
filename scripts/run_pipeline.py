#!/usr/bin/env python3
"""
scripts/run_pipeline.py

End-to-end wrapper for your Helios surrogate pipeline:

1) (optional) Generate Sobol catalog (sim_catalog.parquet)
2) Generate canopy OBJs for a sim range (sim_XXXXXX_field_cropped.obj)
3) Run Helios for the same sim range and save results (CSV + Parquet)

Designed to be restart-safe:
- --skip_existing_obj will skip OBJ generation if canopy already exists
- --skip_existing_results will skip Helios runs if results already exist for a sim_id

Works on OCI (no `module` command) by passing --no_modules.
"""

import os
import json
import argparse
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


def _parse_int_list(s: str) -> List[int]:
    """
    Parse "0" or "0,1,2" into [0] or [0,1,2].
    """
    s = s.strip()
    if not s:
        return [0]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def main():
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: catalog -> canopy OBJs -> Helios runs (results CSV/Parquet)"
    )

    # Overall sim range
    p.add_argument("--start_id", type=int, default=0, help="First simulation_id (inclusive).")
    p.add_argument("--n_sims", type=int, default=10, help="Number of sims to run from start_id.")
    p.add_argument("--end_id", type=int, default=None,
                   help="End simulation_id (exclusive). If set, overrides --n_sims.")

    # Catalog
    p.add_argument("--catalog", type=str, default="catalog/sim_catalog.parquet")
    p.add_argument("--ensure_catalog", action="store_true",
                   help="Create catalog if missing/empty. Uses --catalog_n_sims and --lon_deg.")
    p.add_argument("--catalog_n_sims", type=int, default=10,
                   help="Catalog size to generate when --ensure_catalog is set.")
    p.add_argument("--lon_deg", type=float, default=93.62,
                   help="Fixed longitude used for catalog generation (Ames default in your workflow).")

    # Geometry / canopy generation
    p.add_argument("--base_json", type=str, required=True, help="Base NURBS JSON template path.")
    p.add_argument("--canopy_dir", type=str, default="canopies/test_10")
    p.add_argument("--row_spacing", type=float, default=0.76)
    p.add_argument("--plant_spacing", type=float, default=0.20)
    p.add_argument("--rotation_csv", type=str, default=None)
    p.add_argument("--skip_existing_obj", action="store_true",
                   help="Skip OBJ generation if canopy OBJ already exists.")

    # Helios run
    p.add_argument("--ini_template", type=str, required=True,
                   help="Base Helios INI template to copy per sim.")
    p.add_argument("--config_out_dir", type=str, default="configs/test_10")
    p.add_argument("--utc_offset", type=int, default=5)
    p.add_argument("--helios_build_dir", type=str, required=True,
                   help="Directory where the Helios executable is run from (contains ./par_parallel).")

    # GPUs / modules
    p.add_argument("--device_ids", type=str, default="0",
                   help='Comma-separated GPU ids to round-robin, e.g. "0" or "0,1,2,3".')
    p.add_argument("--no_modules", action="store_true",
                   help="Disable module loading (recommended on OCI where `module` is unavailable).")
    p.add_argument("--modules", type=str, nargs="*", default=None,
                   help="Optional modules to load (HPC only).")

    # Results
    p.add_argument("--results_csv", type=str, default="results/test_10_results.csv")
    p.add_argument("--results_parquet", type=str, default="results/test_10_results.parquet")
    p.add_argument("--skip_existing_results", action="store_true",
                   help="If results file exists, skip sims already present with SUCCESS status.")

    args = p.parse_args()

    # Resolve sim range
    if args.end_id is None:
        start_id = args.start_id
        end_id = args.start_id + args.n_sims
    else:
        start_id = args.start_id
        end_id = args.end_id

    if end_id <= start_id:
        raise SystemExit(f"Invalid range: start_id={start_id}, end_id={end_id}")

    device_ids = _parse_int_list(args.device_ids)
    if not device_ids:
        device_ids = [0]

    # Import your modules (local imports to match your repo layout)
    # 1) catalog generator
    from generate_catalog import main as generate_catalog_main
    # 2) canopy generator
    from generate_canopy_obj import load_catalog, generate_one_canopy
    # 3) helios runner
    from run_helios_single import run_helios_single as run_one_helios

    # Ensure catalog exists if requested
    if args.ensure_catalog:
        if (not os.path.exists(args.catalog)) or os.path.getsize(args.catalog) == 0:
            _ensure_dir(args.catalog)
            generate_catalog_main(
                n_sims=int(args.catalog_n_sims),
                out_path=args.catalog,
                lon_deg_fixed=float(args.lon_deg),
            )

    if not os.path.exists(args.catalog):
        raise SystemExit(f"Catalog not found: {args.catalog}")

    # Load catalog once
    df = load_catalog(args.catalog)
    max_id = int(df["simulation_id"].max())
    if end_id - 1 > max_id:
        raise SystemExit(
            f"Requested end_id={end_id} exceeds max simulation_id in catalog ({max_id}). "
            f"Either generate a bigger catalog or lower end_id."
        )

    # Load existing results if requested
    existing_success = set()
    if args.skip_existing_results and os.path.exists(args.results_parquet):
        try:
            prev = pd.read_parquet(args.results_parquet)
            if "simulation_id" in prev.columns and "status" in prev.columns:
                prev_ok = prev.loc[prev["status"] == "SUCCESS", "simulation_id"].astype(int).tolist()
                existing_success = set(prev_ok)
        except Exception:
            # If file is unreadable, ignore and overwrite later.
            existing_success = set()

    os.makedirs(args.canopy_dir, exist_ok=True)
    os.makedirs(args.config_out_dir, exist_ok=True)
    _ensure_dir(args.results_csv)
    _ensure_dir(args.results_parquet)

    results: List[Dict] = []

    # Pipeline loop
    for k, sim_id in enumerate(range(start_id, end_id)):
        # Skip if already succeeded
        if sim_id in existing_success:
            continue

        # Round-robin GPU selection
        device_id = device_ids[k % len(device_ids)]

        # (A) Ensure canopy OBJ exists (generate if missing)
        geom_status = generate_one_canopy(
            sim_id=sim_id,
            base_json_path=args.base_json,
            out_dir=args.canopy_dir,
            df=df,
            row_spacing=args.row_spacing,
            plant_spacing=args.plant_spacing,
            rotation_csv_path=args.rotation_csv,
            skip_existing=args.skip_existing_obj,
        )

        if geom_status.get("status") not in ("SUCCESS", "SKIPPED"):
            out = {
                "simulation_id": sim_id,
                "status": "FAILED",
                "stage": "GEOMETRY",
                "error": geom_status.get("error", "unknown_geometry_error"),
                "net_PAR": np.nan,
                "obj_path": geom_status.get("obj_path", ""),
                "config_path": "",
                "device_id": device_id,
            }
            print(json.dumps(out, indent=2))
            results.append(out)
            continue

        # (B) Run Helios (this will also ensure per-sim INI gets created/updated)
        run_out = run_one_helios(
            sim_id=sim_id,
            device_id=device_id,
            base_config_template_path=args.ini_template,
            canopy_dir=args.canopy_dir,
            config_out_dir=args.config_out_dir,
            catalog_path=args.catalog,
            base_json_path=args.base_json,
            utc_offset=args.utc_offset,
            row_spacing=args.row_spacing,
            plant_spacing=args.plant_spacing,
            rotation_csv_path=args.rotation_csv,
            skip_existing_obj=True,  # we just ensured it; avoid redoing
            helios_build_dir=args.helios_build_dir,
            modules=args.modules,
            use_modules=(not args.no_modules),
        )

        # Add a couple useful fields for downstream tracking
        run_out["device_id"] = device_id
        run_out["stage"] = "HELIOS"

        print(json.dumps(run_out, indent=2))
        results.append(run_out)

    # Save results
    if results:
        out_df = pd.DataFrame(results)

        # Merge with previous parquet (if exists) so reruns append rather than overwrite
        if os.path.exists(args.results_parquet):
            try:
                prev = pd.read_parquet(args.results_parquet)
                out_df = pd.concat([prev, out_df], ignore_index=True)
                # Drop duplicates, keep last occurrence
                out_df = out_df.sort_values(["simulation_id"]).drop_duplicates(
                    subset=["simulation_id"], keep="last"
                )
            except Exception:
                # If prior file is corrupted, just write new
                pass

        out_df.to_csv(args.results_csv, index=False)
        out_df.to_parquet(args.results_parquet, index=False)

        print(f"\nSaved results:")
        print(f"  CSV:     {args.results_csv}")
        print(f"  Parquet: {args.results_parquet}")
    else:
        print("No new simulations were run (everything was already complete or skipped).")


if __name__ == "__main__":
    main()
