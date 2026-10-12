"""
tools/profiler_tools.py
--------------------------
Core data-profiling logic, exposed as a LangChain @tool so the Data
Profiler ReAct agent can invoke it.
"""

import json

import pandas as pd
from langchain_core.tools import tool


def _infer_type(col_name: str, pct_numeric: float) -> str:
    lowered = col_name.lower()
    if "date" in lowered:
        return "date"
    if pct_numeric > 0.7:
        return "numeric"
    return "categorical_or_text"


def profile_dataset(raw_csv_path: str, output_profile_path: str) -> dict:
    """Reads the raw CSV, computes a structure/quality profile, writes JSON, returns dict."""
    df = pd.read_csv(raw_csv_path, dtype=str)

    profile = {
        "source_file": str(raw_csv_path),
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicate_row_count": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum() + (series.astype(str).str.strip() == "").sum())
        non_null = series.dropna()
        non_null = non_null[non_null.astype(str).str.strip() != ""]

        numeric_coerced = pd.to_numeric(non_null, errors="coerce")
        pct_numeric = float(numeric_coerced.notna().mean()) if len(non_null) else 0.0

        col_profile = {
            "null_count": null_count,
            "null_pct": round(null_count / len(df) * 100, 2) if len(df) else 0.0,
            "distinct_count": int(non_null.nunique()),
            "sample_values": non_null.drop_duplicates().head(5).tolist(),
            "likely_type": _infer_type(col, pct_numeric),
        }
        if col_profile["likely_type"] == "numeric":
            col_profile["unparsable_sample_values"] = (
                non_null[numeric_coerced.isna()].drop_duplicates().head(5).tolist()
            )
            col_profile["pct_parseable_as_numeric"] = round(pct_numeric * 100, 2)

        profile["columns"][col] = col_profile

    with open(output_profile_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)

    return profile


def rule_based_narrative(profile: dict) -> str:
    """Deterministic fallback narrative used when no LLM is configured."""
    issues = []
    for col, p in profile.get("columns", {}).items():
        if p.get("null_pct", 0) > 0:
            issues.append(f"{col} has {p['null_pct']}% missing values")
        if p.get("unparsable_sample_values"):
            issues.append(f"{col} contains non-numeric values e.g. {p['unparsable_sample_values']}")
    if profile.get("duplicate_row_count", 0) > 0:
        issues.append(f"{profile['duplicate_row_count']} exact duplicate rows detected")

    return (
        f"Raw file has {profile.get('row_count', 0)} rows and {profile.get('column_count', 0)} columns. "
        + ("Key issues: " + "; ".join(issues) + "." if issues else "No major issues detected.")
    )


@tool
def profile_raw_data(raw_csv_path: str, output_profile_path: str) -> str:
    """
    Profile the raw sales CSV at raw_csv_path: compute row/column counts,
    null rates, distinct value counts, sample malformed values, and
    duplicate-row counts. Writes the profile as JSON to output_profile_path
    and returns it as a JSON string.
    """
    profile = profile_dataset(raw_csv_path, output_profile_path)
    return json.dumps(profile, default=str)
