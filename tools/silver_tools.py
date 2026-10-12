"""
tools/silver_tools.py
------------------------
Tool the Silver layer execution uses. Calls into the deterministic
transform_engine for all cleansing operations.
"""

import json

import pandas as pd
from langchain_core.tools import tool

from utils import transform_engine


def run_silver(bronze_parquet_path: str, plan: dict, output_parquet_path: str) -> dict:
    df = pd.read_parquet(bronze_parquet_path, engine="pyarrow")
    silver_plan = plan.get("silver", {})
    silver_df, stats = transform_engine.apply_silver_plan(df, silver_plan)
    silver_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    return stats


@tool
def run_silver_transform(bronze_parquet_path: str, plan_path: str, output_parquet_path: str) -> str:
    """
    Executes the Silver-layer rules from the STTM plan JSON file at
    plan_path against the Bronze Parquet file at bronze_parquet_path:
    standardizes dates and currency, deduplicates, flags returns, drops
    truly unparsable rows, and fills missing values. Writes the result as
    a Parquet file to output_parquet_path. Returns a summary string with
    row-level stats.
    """
    with open(plan_path) as f:
        plan = json.load(f)
    stats = run_silver(bronze_parquet_path, plan, output_parquet_path)
    return (
        f"Silver transform complete: rows_in={stats['rows_in']}, rows_out={stats['rows_out']}, "
        f"duplicates_removed={stats['duplicates_removed']}, "
        f"rows_dropped_unparsable={stats['rows_dropped_unparsable']}, "
        f"returns_flagged={stats['returns_flagged']}. "
        f"Written to {output_parquet_path}."
    )
