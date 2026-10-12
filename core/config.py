import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
PROFILES_DIR = DATA_DIR / "profiles"
STTM_DIR = DATA_DIR / "sttm"
BRONZE_DIR = DATA_DIR / "bronze_layer"
SILVER_DIR = DATA_DIR / "silver_layer"
GOLD_DIR = DATA_DIR / "gold_layer"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"
CHROMA_DIR = BASE_DIR / ".chroma"
RUNS_DIR = DATA_DIR / "runs"  # per-run isolated directories (run_manager.py)

# LLM Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

# Groq Configuration (preferred - higher free tier limits)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Use Groq if available, else fall back to Gemini
_env_llm = os.getenv("LLM_PROVIDER", "").strip().lower()
if _env_llm in ("groq", "gemini"):
    LLM_PROVIDER = _env_llm
else:
    LLM_PROVIDER = "groq" if GROQ_API_KEY else "gemini"


def llm_configured() -> bool:
    """True if at least one LLM provider has an API key configured."""
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return True
    if GOOGLE_API_KEY:
        return True
    return False


# Hardcoded retail business requirements used as default context for STTM generation
BUSINESS_REQUIREMENTS = (
    "Analyse retail sales performance across stores, product categories, and time periods. "
    "Identify top-performing stores and categories by revenue. "
    "Flag return patterns and high-discount items. "
    "Provide actionable recommendations for store managers and merchandising teams."
)

# Ensure directories exist
for d in [LANDING_DIR, PROFILES_DIR, STTM_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR,
          REPORTS_DIR, AUDIT_DIR, CHROMA_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
