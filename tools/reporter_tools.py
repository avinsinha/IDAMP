"""
tools/reporter_tools.py
--------------------------
Tools and helpers for the Reporter Agent:
- query_gold_sql: a LangChain @tool letting the agent run read-only DuckDB
  SQL queries against the Gold Parquet table.
- Deterministic KPI computation and Plotly chart generation (kept outside
  the agent loop for reliability).
- HTML report assembly combining the LLM narrative with embedded charts.
"""

import duckdb
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

METRIC_COLUMNS = {
    "total_revenue", "total_quantity", "order_count",
    "avg_order_value", "return_count", "returns_value",
}


# ---------------------------------------------------------------------------
# Agent-facing SQL exploration tool
# ---------------------------------------------------------------------------

@tool
def query_gold_sql(sql: str, gold_parquet_path: str) -> str:
    """
    Runs a read-only DuckDB SQL query against the Gold layer Parquet table.
    The table is available in the query as `gold`. Run "DESCRIBE gold" or
    "SELECT * FROM gold LIMIT 3" first if unsure what columns exist.
    Returns up to 50 result rows as a JSON string.
    """
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "attach", "copy"]
    if any(kw in sql.lower() for kw in forbidden):
        return "ERROR: only read-only SELECT queries are permitted."

    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW gold AS SELECT * FROM read_parquet('{gold_parquet_path}')")
    try:
        result_df = con.execute(sql).fetchdf()
    except Exception as e:
        return f"ERROR running query: {e}"
    finally:
        con.close()

    return result_df.head(50).to_json(orient="records")


# ---------------------------------------------------------------------------
# Schema introspection helpers
# ---------------------------------------------------------------------------

def _get_columns(con) -> list:
    return [row[0] for row in con.execute("DESCRIBE gold").fetchall()]


def _find_column(columns: list, *keywords: str):
    """Returns the first non-metric column whose name contains any keyword."""
    for col in columns:
        if col in METRIC_COLUMNS:
            continue
        if any(kw in col.lower() for kw in keywords):
            return col
    return None


# ---------------------------------------------------------------------------
# Deterministic KPI computation
# ---------------------------------------------------------------------------

def compute_kpis(gold_parquet_path: str) -> dict:
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW gold AS SELECT * FROM read_parquet('{gold_parquet_path}')")

    columns = _get_columns(con)
    category_col = _find_column(columns, "category")
    store_col = _find_column(columns, "store")
    date_col = _find_column(columns, "date", "sale")

    totals = con.execute(
        "SELECT SUM(total_revenue) AS total_revenue, SUM(order_count) AS total_orders, "
        "SUM(returns_value) AS total_returns_value, COUNT(*) AS row_count FROM gold"
    ).fetchdf().iloc[0]

    by_category_df = None
    if category_col:
        by_category_df = con.execute(
            f'SELECT "{category_col}" AS dimension, SUM(total_revenue) AS revenue FROM gold '
            f'GROUP BY "{category_col}" ORDER BY revenue DESC LIMIT 10'
        ).fetchdf()

    by_store_df = None
    if store_col:
        by_store_df = con.execute(
            f'SELECT "{store_col}" AS dimension, SUM(total_revenue) AS revenue FROM gold '
            f'GROUP BY "{store_col}" ORDER BY revenue DESC LIMIT 10'
        ).fetchdf()

    by_date_df = None
    if date_col:
        by_date_df = con.execute(
            f'SELECT "{date_col}" AS dimension, SUM(total_revenue) AS revenue FROM gold '
            f'GROUP BY "{date_col}" ORDER BY "{date_col}"'
        ).fetchdf()

    con.close()

    total_orders = int(totals["total_orders"] or 0)
    total_revenue = float(totals["total_revenue"] or 0.0)

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "avg_order_value": round(total_revenue / total_orders, 2) if total_orders else 0.0,
        "total_returns_value": round(float(totals["total_returns_value"] or 0.0), 2),
        "row_count_gold": int(totals["row_count"]),
        "top_categories": dict(zip(by_category_df["dimension"], by_category_df["revenue"].round(2))) if by_category_df is not None else {},
        "top_stores": dict(zip(by_store_df["dimension"], by_store_df["revenue"].round(2))) if by_store_df is not None else {},
        "category_label": category_col or "Category",
        "store_label": store_col or "Store",
        "date_label": date_col or "Date",
        "_by_category_df": by_category_df,
        "_by_store_df": by_store_df,
        "_by_date_df": by_date_df,
    }


# ---------------------------------------------------------------------------
# Deterministic Plotly chart generation
# ---------------------------------------------------------------------------

def build_charts(kpis: dict) -> dict:
    """Returns chart_name -> HTML div string. Keys present only for existing dimensions."""
    charts = {}
    first_chart = True

    def _include_js():
        nonlocal first_chart
        val = "cdn" if first_chart else False
        first_chart = False
        return val

    if kpis["_by_category_df"] is not None:
        fig_cat = px.bar(
            kpis["_by_category_df"], x="dimension", y="revenue",
            title=f"Revenue by {kpis['category_label']}",
            labels={"revenue": "Revenue ($)", "dimension": kpis["category_label"]},
            color="dimension",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig_cat.update_layout(showlegend=False)
        charts["revenue_by_category"] = fig_cat.to_html(full_html=False, include_plotlyjs=_include_js())

        fig_pie = px.pie(
            kpis["_by_category_df"], names="dimension", values="revenue",
            title=f"Revenue Share by {kpis['category_label']}",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig_pie.update_traces(marker=dict(
            colors=px.colors.qualitative.Set2, line=dict(color="white", width=1)
        ))
        charts["category_share"] = fig_pie.to_html(full_html=False, include_plotlyjs=_include_js())

    if kpis["_by_store_df"] is not None:
        fig_store = px.bar(
            kpis["_by_store_df"], x="dimension", y="revenue",
            title=f"Revenue by {kpis['store_label']}",
            labels={"revenue": "Revenue ($)", "dimension": kpis["store_label"]},
            color="dimension",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_white",
        )
        fig_store.update_layout(showlegend=False)
        charts["revenue_by_store"] = fig_store.to_html(full_html=False, include_plotlyjs=_include_js())

    if kpis["_by_date_df"] is not None:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=kpis["_by_date_df"]["dimension"], y=kpis["_by_date_df"]["revenue"],
            mode="lines+markers", name="Revenue",
            line=dict(color="#2c6e49", width=2),
            marker=dict(color="#2c6e49", size=6),
        ))
        fig_trend.update_layout(
            title="Revenue Trend Over Time",
            xaxis_title=kpis["date_label"],
            yaxis_title="Revenue ($)",
            template="plotly_white",
        )
        charts["revenue_trend"] = fig_trend.to_html(full_html=False, include_plotlyjs=_include_js())

    if not charts:
        fig = go.Figure()
        fig.add_annotation(
            text="No dimension columns available to chart for this run's Gold grain.",
            showarrow=False, font=dict(size=14),
        )
        fig.update_layout(title="Revenue Overview", xaxis_visible=False, yaxis_visible=False)
        charts["overview"] = fig.to_html(full_html=False, include_plotlyjs="cdn")

    return charts


# ---------------------------------------------------------------------------
# HTML report assembly
# ---------------------------------------------------------------------------

def assemble_html_report(narrative_text: str, kpis: dict, charts: dict, run_id: str) -> str:
    narrative_html = "".join(
        f"<p>{line}</p>" for line in narrative_text.strip().split("\n") if line.strip()
    )

    kpi_cards = f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Total Revenue</div><div class="kpi-value">${kpis['total_revenue']:,.2f}</div></div>
      <div class="kpi-card"><div class="kpi-label">Total Orders</div><div class="kpi-value">{kpis['total_orders']:,}</div></div>
      <div class="kpi-card"><div class="kpi-label">Avg Order Value</div><div class="kpi-value">${kpis['avg_order_value']:,.2f}</div></div>
      <div class="kpi-card"><div class="kpi-label">Returns Value</div><div class="kpi-value">${kpis['total_returns_value']:,.2f}</div></div>
    </div>
    """

    chart_sections = ""
    section_titles = {
        "revenue_by_category": f"Revenue by {kpis.get('category_label', 'Category')}",
        "revenue_by_store": f"Revenue by {kpis.get('store_label', 'Store')}",
        "revenue_trend": "Revenue Trend",
        "category_share": f"{kpis.get('category_label', 'Category')} Revenue Share",
        "overview": "Overview",
    }
    for key, html_snippet in charts.items():
        title = section_titles.get(key, key.replace("_", " ").title())
        chart_sections += f'<h2>{title}</h2>\n<div class="chart-container">{html_snippet}</div>\n'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sales Report — Gold Layer Analytics</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1a1a1a; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #2c6e49; padding-bottom: 8px; }}
  h2 {{ color: #2c6e49; margin-top: 40px; }}
  .kpi-grid {{ display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }}
  .kpi-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px 24px; min-width: 160px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .kpi-label {{ font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 24px; font-weight: 600; color: #1a1a1a; margin-top: 4px; }}
  .chart-container {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 20px 0; }}
  .run-id {{ color: #999; font-size: 12px; }}
  .narrative p {{ line-height: 1.6; }}
</style>
</head>
<body>
<h1>Sales Report — Gold Layer Analytics</h1>
<p class="run-id">Run ID: {run_id}</p>

<h2>Executive Summary</h2>
<div class="narrative">{narrative_html}</div>

<h2>Key Metrics</h2>
{kpi_cards}

{chart_sections}

</body>
</html>
"""
