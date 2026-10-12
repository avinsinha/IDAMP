"""
main_cli.py
-------------
CLI runner for the IDAMP pipeline (useful for testing/automation outside
the Streamlit UI). Generates synthetic messy data, runs the full medallion
pipeline without HITL gates, and prints a summary.

Usage:
    python main_cli.py                    # synthetic data, default business intent
    python main_cli.py --rows 1000
    python main_cli.py --csv path/to.csv --intent "Which stores drive the most revenue?"
"""

import argparse
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

import run_manager
import llm_provider
from utils.data_generator import generate_raw_sales_csv
from tools.profiler_tools import profile_dataset, rule_based_narrative
from tools.sttm_tools import DEFAULT_PLAN, save_plan
from tools.bronze_tools import run_bronze
from tools.silver_tools import run_silver
from tools.gold_tools import run_gold
from tools.reporter_tools import compute_kpis, build_charts, assemble_html_report

BUSINESS_REQUIREMENTS = (
    "Analyse retail sales performance across stores, product categories, and time periods. "
    "Identify top-performing stores and categories by revenue. Flag return patterns. "
    "Provide actionable recommendations for the business."
)


def _generate_sttm_plan(paths, profile: dict) -> dict:
    """Use LLM if available, else return the deterministic DEFAULT_PLAN."""
    model = llm_provider.get_chat_model()
    if model is None:
        logging.getLogger("main_cli").info("LLM unavailable — using rule-based DEFAULT_PLAN.")
        plan = json.loads(json.dumps(DEFAULT_PLAN))
        plan["_meta"] = {"generated_by": "rule_based_fallback"}
        save_plan(plan, str(paths.sttm_json))
        return plan

    from langchain.agents import create_agent  # noqa: F401 — replaced below
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    from tools.sttm_tools import save_sttm_plan

    SYSTEM_PROMPT = (
        "You are a senior data engineer designing a Medallion Architecture "
        "(Bronze/Silver/Gold) transformation plan from a data profile and "
        "business requirements. Base column names strictly on those present in "
        "the profile. Call save_sttm_plan exactly once with your finished plan."
    )

    agent = create_react_agent(model, tools=[save_sttm_plan], prompt=SYSTEM_PROMPT)
    user_msg = (
        f"BUSINESS REQUIREMENTS:\n{BUSINESS_REQUIREMENTS}\n\n"
        f"DATA PROFILE:\n{json.dumps(profile, indent=2, default=str)}\n\n"
        f"Design the transformation plan and save it to '{paths.sttm_json}'."
    )
    agent.invoke({"messages": [HumanMessage(content=user_msg)]})

    with open(paths.sttm_json) as f:
        plan = json.load(f)
    plan.setdefault("_meta", {})["generated_by"] = "llm"
    save_plan(plan, str(paths.sttm_json))
    return plan


def _generate_report_narrative(kpis: dict) -> str:
    """Use LLM reporter agent if available, else build a templated narrative."""
    model = llm_provider.get_chat_model()
    if model is None:
        cats = ", ".join(f"{k} (${v:,.2f})" for k, v in list(kpis["top_categories"].items())[:3])
        stores = ", ".join(f"{k} (${v:,.2f})" for k, v in list(kpis["top_stores"].items())[:3])
        return (
            f"Pipeline processed data through Bronze, Silver, and Gold layers, producing "
            f"{kpis['row_count_gold']} aggregated analytics rows. "
            f"Total net revenue: ${kpis['total_revenue']:,.2f} across {kpis['total_orders']} orders. "
            f"Average order value: ${kpis['avg_order_value']:,.2f}. "
            f"Returns totalled ${kpis['total_returns_value']:,.2f}.\n"
            f"Top categories: {cats}.\nTop stores: {stores}.\n"
            "Recommendation: replicate top-category strategies across underperforming segments."
        )

    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import HumanMessage
    from tools.reporter_tools import query_gold_sql

    REPORTER_PROMPT = (
        "You are a retail analytics reporting agent. Use query_gold_sql to explore the "
        "Gold table, then write a concise executive summary with 2-3 actionable recommendations. "
        "Plain prose only — no markdown."
    )
    agent = create_react_agent(model, tools=[query_gold_sql], prompt=REPORTER_PROMPT)
    result = agent.invoke({"messages": [HumanMessage(content=BUSINESS_REQUIREMENTS)]})
    return result["messages"][-1].content or "Report generation complete."


def main():
    parser = argparse.ArgumentParser(description="Run the IDAMP pipeline (CLI mode, no HITL).")
    parser.add_argument("--rows", type=int, default=600, help="Synthetic rows to generate (ignored if --csv given).")
    parser.add_argument("--csv", type=str, default=None, help="Path to an existing raw CSV (skips generation).")
    parser.add_argument("--intent", type=str, default=BUSINESS_REQUIREMENTS, help="Business intent string.")
    args = parser.parse_args()

    logger = logging.getLogger("main_cli")
    paths = run_manager.new_run()
    print(f"\nRun ID:  {paths.run_id}")
    print(f"LLM:     {'configured' if llm_provider.is_configured() else 'not configured — rule-based fallback'}")
    print(f"Output:  {paths.root}\n")

    # Step 1: data
    if args.csv:
        import shutil
        shutil.copy(args.csv, paths.raw_csv)
        print(f"Using supplied CSV: {args.csv}")
    else:
        print(f"Generating {args.rows} synthetic messy rows...")
        generate_raw_sales_csv(paths.raw_csv, n_rows=args.rows)

    # Step 2: profile
    print("Profiling raw data...")
    profile = profile_dataset(str(paths.raw_csv), str(paths.profile_json))
    print(f"  -> {profile['row_count']} rows, {profile['column_count']} columns, "
          f"{profile['duplicate_row_count']} duplicates")
    print(f"  -> {rule_based_narrative(profile)}\n")

    # Step 3: STTM plan
    print("Generating STTM plan (auto-approved in CLI mode)...")
    plan = _generate_sttm_plan(paths, profile)
    print(f"  -> Plan generated by: {plan.get('_meta', {}).get('generated_by', 'unknown')}\n")

    # Step 4: bronze
    print("Running Bronze transform...")
    bronze_stats = run_bronze(str(paths.raw_csv), plan, str(paths.bronze_parquet))
    print(f"  -> {bronze_stats}\n")

    # Step 5: silver
    print("Running Silver transform...")
    silver_stats = run_silver(str(paths.bronze_parquet), plan, str(paths.silver_parquet))
    print(f"  -> {silver_stats}\n")

    # Step 6: gold
    print("Running Gold transform...")
    gold_stats = run_gold(str(paths.silver_parquet), plan, str(paths.gold_parquet))
    print(f"  -> {gold_stats}\n")

    # Step 7: report
    print("Generating HTML report...")
    kpis = compute_kpis(str(paths.gold_parquet))
    charts = build_charts(kpis)
    narrative = _generate_report_narrative(kpis)
    html_report = assemble_html_report(narrative, kpis, charts, paths.run_id)
    with open(paths.report_html, "w") as f:
        f.write(html_report)

    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Bronze : {bronze_stats['rows']} rows, {bronze_stats['columns']} columns")
    print(f"Silver : rows_in={silver_stats['rows_in']}, rows_out={silver_stats['rows_out']}, "
          f"dupes_removed={silver_stats['duplicates_removed']}")
    print(f"Gold   : {gold_stats['rows']} aggregated rows")
    print(f"KPIs   : revenue=${kpis['total_revenue']:,.2f}, orders={kpis['total_orders']}")
    print()
    print(f"Artifacts: {paths.root}")
    print(f"  Profile : {paths.profile_json}")
    print(f"  STTM    : {paths.sttm_json}")
    print(f"  Bronze  : {paths.bronze_parquet}")
    print(f"  Silver  : {paths.silver_parquet}")
    print(f"  Gold    : {paths.gold_parquet}")
    print(f"  Report  : {paths.report_html}")
    print("=" * 70)


if __name__ == "__main__":
    main()
