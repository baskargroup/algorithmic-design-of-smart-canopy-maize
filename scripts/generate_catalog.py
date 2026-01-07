# scripts/generate_catalog.py

import numpy as np
import pandas as pd
from scipy.stats import qmc
import os
import argparse

NUM_LEAVES = 12
D_GEOM = 1 + 7 * NUM_LEAVES        # 85
D_TOTAL = D_GEOM + 2               # + lat, lon = 87 (kept unchanged)

# Fixed longitude: Ames, IA
AMES_LON_DEG = 93.62


def generate_sobol_points(n_sims: int, d: int) -> np.ndarray:
    sampler = qmc.Sobol(d=d, scramble=True)

    # Sobol performs best with power-of-2 samples;
    # we generate enough and truncate.
    m = int(np.ceil(np.log2(n_sims)))
    sobol_raw = sampler.random_base2(m=m)  # shape: (2**m, d)
    return sobol_raw[:n_sims, :]


def map_to_range(x, lo, hi):
    return lo + x * (hi - lo)


def main(n_sims: int, out_path: str, lon_deg_fixed: float = AMES_LON_DEG):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Generate Sobol samples in [0,1]
    sobol = generate_sobol_points(n_sims, D_TOTAL)

    # Split geometry vs latitude/longitude slots
    sobol_geom = sobol[:, :D_GEOM]
    sobol_lat  = sobol[:, D_GEOM + 0]
    # sobol_lon = sobol[:, D_GEOM + 1]  # intentionally unused

    # Latitude varies
    lat_deg = map_to_range(sobol_lat, 7.0, 48.0)   # 7–48°N

    # Longitude fixed at Ames
    lon_deg = np.full(n_sims, lon_deg_fixed, dtype=float)

    records = []

    for sim_id in range(n_sims):
        row = {
            "simulation_id": sim_id,
            "lat_deg": float(lat_deg[sim_id]),
            "lon_deg": float(lon_deg[sim_id]),
        }

        # Geometry dimensions (expected by canopy_from_sobol.py)
        # [stalk, interleaf[L], length[L], width[L],
        #  theta[L], phi[L], curvature[L], twist[L]]
        g = sobol_geom[sim_id, :]

        row["stalk_raw"] = g[0]

        idx = 1
        for i in range(NUM_LEAVES):
            row[f"interleaf_raw_{i}"] = g[idx]; idx += 1
        for i in range(NUM_LEAVES):
            row[f"length_raw_{i}"]    = g[idx]; idx += 1
        for i in range(NUM_LEAVES):
            row[f"width_raw_{i}"]     = g[idx]; idx += 1
        for i in range(NUM_LEAVES):
            row[f"theta_raw_{i}"]     = g[idx]; idx += 1
        for i in range(NUM_LEAVES):
            row[f"phi_raw_{i}"]       = g[idx]; idx += 1
        for i in range(NUM_LEAVES):
            row[f"curv_raw_{i}"]      = g[idx]; idx += 1
        for i in range(NUM_LEAVES):
            row[f"twist_raw_{i}"]     = g[idx]; idx += 1

        records.append(row)

    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"Saved catalog with {n_sims} sims to {out_path}")
    print(f"Longitude fixed at {lon_deg_fixed}° (Ames, IA)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sobol catalog for Helios surrogate pipeline.")
    parser.add_argument("--n_sims", type=int, required=True)
    parser.add_argument("--out_path", type=str, default="catalog/sim_catalog.parquet")
    parser.add_argument("--lon_deg", type=float, default=AMES_LON_DEG)
    args = parser.parse_args()

    main(n_sims=args.n_sims, out_path=args.out_path, lon_deg_fixed=args.lon_deg)