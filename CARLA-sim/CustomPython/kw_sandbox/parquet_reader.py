import argparse
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def sql_escape_literal(s: str) -> str:
    return s.replace("'", "''")


def get_columns(con: duckdb.DuckDBPyConnection, parquet_path: str) -> set:
    # Return a set of column names in the Parquet file
    escaped = sql_escape_literal(parquet_path)
    rows = con.execute(
        f"DESCRIBE SELECT * FROM parquet_scan('{escaped}')"
    ).fetchall()
    # column_name is first element in each row
    return {row[0] for row in rows}


def build_distance_expr(cols: set) -> str:
    # If the Parquet has 'distance_ego_npc', we just use that. Otherwise, we compute it from 
    # ego_x, ego_y, npc_x, npc_y.
    if "distance_ego_npc" in cols:
        return "distance_ego_npc"

    required = {"ego_x", "ego_y", "npc_x", "npc_y"}
    if not required.issubset(cols):
        missing = required - cols
        raise ValueError(
            f"Cannot compute distance_ego_npc; missing columns: {missing}"
        )

    # Euclidean distance in the XY plane
    return "sqrt(power(ego_x - npc_x, 2) + power(ego_y - npc_y, 2))"


def make_distance_subquery(parquet_path: str, distance_expr: str) -> str:
    # Build a subquery which can be reused in all our queries (count/min/max/export).
    escaped = sql_escape_literal(parquet_path)
    subquery = f"""
        SELECT
            frame,
            sim_time,
            {distance_expr} AS dist
        FROM parquet_scan('{escaped}')
    """
    return subquery


def compute_summary(
    con: duckdb.DuckDBPyConnection,
    distance_subquery: str,
):
    # Compute and return:
    # - number of valid samples
    # - row with minimum distance
    # - row with maximum distance

    # valid rows where dist >= 0 - negative could mean there is no NPC in the sim
    count_valid = con.execute(
        f"""
        SELECT COUNT(*)
        FROM ({distance_subquery}) AS t
        WHERE dist >= 0
        """
    ).fetchone()[0]

    if count_valid == 0:
        raise ValueError("No valid NPC distance data (all dist < 0 or no NPC present).")

    # Min distance row
    min_row = con.execute(
        f"""
        SELECT frame, sim_time, dist
        FROM ({distance_subquery}) AS t
        WHERE dist >= 0
        ORDER BY dist ASC
        LIMIT 1
        """
    ).fetchone()

    # Max distance row
    max_row = con.execute(
        f"""
        SELECT frame, sim_time, dist
        FROM ({distance_subquery}) AS t
        WHERE dist >= 0
        ORDER BY dist DESC
        LIMIT 1
        """
    ).fetchone()

    return min_row, max_row, int(count_valid)


def export_distances(
    con: duckdb.DuckDBPyConnection,
    distance_subquery: str,
    output_path: str,
):
    # For each frame, export frame, sim time, and ego-npc distances to a new Parquet file 
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrow_table = con.execute(
        f"""
        SELECT
            frame,
            sim_time,
            dist AS distance_ego_npc
        FROM ({distance_subquery}) AS t
        WHERE dist >= 0
        ORDER BY frame
        """
    ).fetch_arrow_table()

    pq.write_table(arrow_table, out.as_posix())
    print(f"Saved distance features to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze CARLA Parquet log with DuckDB (no pandas/numpy) and extract "
            "ego–NPC distance features."
        )
    )
    parser.add_argument("input", help="Path to input Parquet log file.")
    parser.add_argument(
        "--out",
        help="Optional path to output Parquet with [frame, sim_time, distance_ego_npc].",
        default=None,
    )
    args = parser.parse_args()

    parquet_path = Path(args.input)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    con = duckdb.connect()

    # Inspect schema
    cols = get_columns(con, parquet_path.as_posix())

    # Build distance expression and subquery
    distance_expr = build_distance_expr(cols)
    distance_subquery = make_distance_subquery(parquet_path.as_posix(), distance_expr)

    # Compute summary stats
    min_row, max_row, count_valid = compute_summary(con, distance_subquery)

    min_frame, min_time, min_dist = min_row
    max_frame, max_time, max_dist = max_row

    print("\n=== Ego–NPC Distance Summary (DuckDB, no pandas/numpy) ===")
    print(f"Valid samples: {count_valid}")

    print(f"Min distance: {min_dist:.3f} m")
    print(f"  at frame: {int(min_frame)}")
    print(f"  at sim_time: {float(min_time):.3f} s")

    print(f"Max distance: {max_dist:.3f} m")
    print(f"  at frame: {int(max_frame)}")
    print(f"  at sim_time: {float(max_time):.3f} s")

    # export
    if args.out is not None:
        export_distances(con, distance_subquery, args.out)

    con.close()


if __name__ == "__main__":
    main()
