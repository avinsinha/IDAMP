"""
tools/gold_tools.py
----------------------
Tool the Gold layer execution uses. Calls into the deterministic
transform_engine for all aggregation operations.
"""

import json

import pandas as pd
from langchain_core.tools import tool

from utils import transform_engine


def run_gold(silver_parquet_path: str, plan: dict, output_parquet_path: str) -> dict:
    df = pd.read_parquet(silver_parquet_path, engine="pyarrow")
    gold_plan = plan.get("gold", {})
    gold_df = transform_engine.apply_gold_plan(df, gold_plan)
    gold_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    return {"rows": len(gold_df), "columns": len(gold_df.columns)}


@tool
def run_gold_transform(silver_parquet_path: str, plan_path: str, output_parquet_path: str) -> str:
    """
    Executes the Gold-layer rules from the STTM plan JSON file at
    plan_path against the Silver Parquet file at silver_parquet_path:
    aggregates into an analytics-ready table (revenue, quantity, order
    counts, avg order value, returns) at the specified grain. Writes the
    result as a Parquet file to output_parquet_path. Returns a summary
    string.
    """
    with open(plan_path) as f:
        plan = json.load(f)
    result = run_gold(silver_parquet_path, plan, output_parquet_path)
    return (
        f"Gold transform complete: {result['rows']} aggregated rows, "
        f"{result['columns']} columns written to {output_parquet_path}."
    )
