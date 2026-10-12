"""
tools/bronze_tools.py
------------------------
Tool the Bronze layer execution uses. Calls into the deterministic
transform_engine so the LLM never generates or executes arbitrary code.
"""

import json

import pandas as pd
from langchain_core.tools import tool

from utils import transform_engine


def run_bronze(raw_csv_path: str, plan: dict, output_parquet_path: str) -> dict:
    df = pd.read_csv(raw_csv_path, dtype=str)
    bronze_plan = plan.get("bronze", {})
    bronze_df = transform_engine.apply_bronze_plan(df, bronze_plan, source_file_name=str(raw_csv_path))
    bronze_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    return {"rows": len(bronze_df), "columns": len(bronze_df.columns)}


@tool
def run_bronze_transform(raw_csv_path: str, plan_path: str, output_parquet_path: str) -> str:
    """
    Executes the Bronze-layer rules from the STTM plan JSON file at
    plan_path against the raw CSV at raw_csv_path: standardizes column
    names/types and stamps ingestion metadata, preserving raw data
    fidelity. Writes the result as a Parquet file to output_parquet_path.
    Returns a summary string.
    """
    with open(plan_path) as f:
        plan = json.load(f)
    result = run_bronze(raw_csv_path, plan, output_parquet_path)
    return (
        f"Bronze transform complete: {result['rows']} rows, "
        f"{result['columns']} columns written to {output_parquet_path}."
    )
