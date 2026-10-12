"""
utils/transform_engine.py
--------------------------
Deterministic operations that interpret the STTM plan JSON against a
pandas DataFrame. The LLM (via the STTM agents) decides *what* the plan
should contain; these functions decide *how* it gets executed — kept
fixed and auditable rather than having an LLM generate/execute arbitrary
code at runtime.
"""

import re
import logging
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger("transform_engine")


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

def apply_bronze_plan(df: pd.DataFrame, plan: dict, source_file_name: str) -> pd.DataFrame:
    df = df.copy()

    rename_map = plan.get("rename", {}) or {}
    if rename_map:
        df = df.rename(columns=rename_map)

    for col, dtype in (plan.get("cast", {}) or {}).items():
        if col not in df.columns:
            continue
        try:
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception as e:
            logger.warning("Could not cast column %s to %s: %s", col, dtype, e)

    add_columns = plan.get("add_columns", {}) or {}
    if add_columns.get("ingestion_ts"):
        df["ingestion_ts"] = datetime.now(timezone.utc).isoformat()
    if add_columns.get("source_file"):
        df["source_file"] = source_file_name

    return df


# ---------------------------------------------------------------------------
# Silver
# ---------------------------------------------------------------------------

_DATE_TRY_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%m-%d-%Y", "%Y/%m/%d"]


def _parse_messy_date(value):
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    value = str(value).strip()
    for fmt in _DATE_TRY_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if not pd.isna(parsed) else pd.NaT


def _parse_currency(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def apply_silver_plan(df: pd.DataFrame, plan: dict):
    df = df.copy()
    stats = {
        "rows_in": len(df),
        "duplicates_removed": 0,
        "rows_dropped_unparsable": 0,
        "returns_flagged": 0,
        "customer_id_filled": 0,
    }

    date_col = plan.get("standardize_date_column")
    if date_col and date_col in df.columns:
        df[date_col] = df[date_col].apply(_parse_messy_date)

    for col in plan.get("clean_currency_columns", []) or []:
        if col in df.columns:
            df[col] = df[col].apply(_parse_currency)

    for col in plan.get("standardize_text_columns", ["store_name"]):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    qty_col = "quantity" if "quantity" in df.columns else None
    if qty_col:
        df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce")

    if plan.get("handle_negative_quantity") == "flag_as_return" and qty_col:
        df["is_return"] = df[qty_col] < 0
        stats["returns_flagged"] = int(df["is_return"].sum())
    else:
        df["is_return"] = False

    price_col = "unit_price" if "unit_price" in df.columns else None
    required_cols = [c for c in [qty_col, price_col] if c]
    before = len(df)
    if required_cols:
        df = df.dropna(subset=required_cols)
    stats["rows_dropped_unparsable"] = before - len(df)

    fill_map = plan.get("fillna", {}) or {}
    if "customer_id" in df.columns and "customer_id" in fill_map:
        missing_mask = df["customer_id"].isna() | (df["customer_id"].astype(str).str.strip() == "")
        stats["customer_id_filled"] = int(missing_mask.sum())
        df.loc[missing_mask, "customer_id"] = fill_map["customer_id"]

    dedup_keys = plan.get("dedup_keys")
    before = len(df)
    if dedup_keys and all(k in df.columns for k in dedup_keys):
        df = df.drop_duplicates(subset=dedup_keys, keep="first")
    else:
        df = df.drop_duplicates(keep="first")
    stats["duplicates_removed"] = before - len(df)

    stats["rows_out"] = len(df)
    return df.reset_index(drop=True), stats


# ---------------------------------------------------------------------------
# Gold
# ---------------------------------------------------------------------------

def apply_gold_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    df = df.copy()
    group_by = [c for c in plan.get("group_by", []) if c in df.columns]

    quantity = pd.to_numeric(df.get("quantity", 0), errors="coerce").fillna(0)
    unit_price = pd.to_numeric(df.get("unit_price", 0), errors="coerce").fillna(0)
    discount_pct = (
        pd.to_numeric(df.get("discount_pct", 0), errors="coerce").fillna(0)
        if "discount_pct" in df.columns else 0
    )

    df["quantity"] = quantity
    df["unit_price"] = unit_price
    df["line_revenue"] = quantity.abs() * unit_price * (1 - discount_pct / 100)
    df["is_return"] = df.get("is_return", False)

    def agg_group(g: pd.DataFrame) -> pd.Series:
        sales = g[~g["is_return"]]
        returns = g[g["is_return"]]
        id_col = next((c for c in ["transaction_id", "order_id"] if c in g.columns), None)
        order_count = sales[id_col].nunique() if id_col else len(sales)
        total_revenue = sales["line_revenue"].sum()
        return pd.Series({
            "total_revenue": round(total_revenue, 2),
            "total_quantity": int(sales["quantity"].sum()) if not sales.empty else 0,
            "order_count": int(order_count),
            "avg_order_value": round(total_revenue / order_count, 2) if order_count else 0.0,
            "return_count": int(len(returns)),
            "returns_value": round(returns["line_revenue"].sum(), 2) if not returns.empty else 0.0,
        })

    if not group_by:
        group_by = [c for c in ["store_name", "product_category", "sale_date"] if c in df.columns]

    gold = df.groupby(group_by, dropna=False).apply(agg_group).reset_index()
    return gold
