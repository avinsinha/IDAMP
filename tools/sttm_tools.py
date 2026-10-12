"""
tools/sttm_tools.py
----------------------
Tool the STTM Agent uses to submit its finished transformation plan. Using
a "save_sttm_plan" tool call (rather than parsing free-text JSON out of the
final message) is a more robust structured-output pattern for a ReAct agent.
"""

import json

from langchain_core.tools import tool

DEFAULT_PLAN = {
    "bronze": {
        "rename": {},
        "cast": {},
        "add_columns": {"ingestion_ts": True, "source_file": True},
    },
    "silver": {
        "standardize_date_column": "sale_date",
        "clean_currency_columns": ["unit_price"],
        "standardize_text_columns": ["store_name"],
        "handle_negative_quantity": "flag_as_return",
        "dedup_keys": ["transaction_id"],
        "fillna": {"discount_pct": 0},
    },
    "gold": {
        "group_by": ["store_name", "category", "sale_date"],
    },
}

REQUIRED_TOP_LEVEL_KEYS = {"bronze", "silver", "gold"}


def validate_plan(plan: dict) -> list:
    """Returns a list of validation error strings (empty if valid)."""
    errors = []
    missing = REQUIRED_TOP_LEVEL_KEYS - set(plan.keys())
    if missing:
        errors.append(f"Missing required top-level keys: {sorted(missing)}")
    return errors


def save_plan(plan: dict, output_plan_path: str) -> None:
    with open(output_plan_path, "w") as f:
        json.dump(plan, f, indent=2)


@tool
def save_sttm_plan(plan_json: str, output_plan_path: str) -> str:
    """
    Validates and saves the finished Bronze/Silver/Gold transformation plan.
    plan_json must be a JSON string with this exact top-level shape:
    {"bronze": {"rename": {}, "cast": {}, "add_columns": {"ingestion_ts": true, "source_file": true}},
     "silver": {"standardize_date_column": "<col>", "clean_currency_columns": ["<col>"],
                "standardize_text_columns": ["<col>"], "handle_negative_quantity": "flag_as_return",
                "dedup_keys": ["<col>"], "fillna": {"<col>": "<value>"}},
     "gold": {"group_by": ["<col>", ...]}}
    Call this exactly once with your final plan. Returns a confirmation
    message, or an error message if validation fails (fix and call again).
    """
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as e:
        return f"ERROR: plan_json was not valid JSON: {e}. Please fix and call save_sttm_plan again."

    errors = validate_plan(plan)
    if errors:
        return f"ERROR: plan validation failed: {errors}. Please fix and call save_sttm_plan again."

    save_plan(plan, output_plan_path)
    return "Plan saved successfully."
