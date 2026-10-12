"""
run_manager.py
---------------
Per-run isolated directory management for IDAMP.

Each pipeline run gets its own subdirectory under data/runs/<run_id>/
with all artifacts isolated. Prevents runs from overwriting each other.
The existing shared directories (bronze_layer/, silver_layer/, etc.) remain
for backwards compatibility with the Streamlit UI.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.config import DATA_DIR, REPORTS_DIR


RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunPaths:
    """Resolved paths for a single pipeline run, all isolated under data/runs/<run_id>/."""

    run_id: str
    root: Path = field(init=False)
    raw: Path = field(init=False)
    bronze: Path = field(init=False)
    silver: Path = field(init=False)
    gold: Path = field(init=False)
    profiles: Path = field(init=False)
    sttm: Path = field(init=False)
    reports: Path = field(init=False)

    def __post_init__(self):
        self.root = RUNS_DIR / self.run_id
        self.raw = self.root / "raw"
        self.bronze = self.root / "bronze"
        self.silver = self.root / "silver"
        self.gold = self.root / "gold"
        self.profiles = self.root / "profiles"
        self.sttm = self.root / "sttm"
        self.reports = self.root / "reports"
        for d in [self.raw, self.bronze, self.silver, self.gold, self.profiles, self.sttm, self.reports]:
            d.mkdir(parents=True, exist_ok=True)

    # --- Convenience path properties ---

    @property
    def raw_csv(self) -> Path:
        return self.raw / "raw_sales.csv"

    @property
    def bronze_parquet(self) -> Path:
        return self.bronze / "bronze.parquet"

    @property
    def silver_parquet(self) -> Path:
        return self.silver / "silver.parquet"

    @property
    def gold_parquet(self) -> Path:
        return self.gold / "gold.parquet"

    @property
    def profile_json(self) -> Path:
        return self.profiles / "profile.json"

    @property
    def sttm_json(self) -> Path:
        return self.sttm / "sttm_plan.json"

    @property
    def report_html(self) -> Path:
        return self.reports / "sales_report.html"

    @property
    def run_log(self) -> Path:
        return self.root / "run.log"


def new_run() -> RunPaths:
    """Create and return a fresh RunPaths for a new pipeline run."""
    run_id = str(uuid.uuid4())[:8]
    return RunPaths(run_id=run_id)
