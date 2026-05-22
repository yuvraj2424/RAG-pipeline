# Centralized configuration — all hardcoded values live here
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = DATA_DIR / "rag_db"
SOURCE_DIR = ROOT_DIR / "source"

# ── Azure AI Foundry (Claude models) ────────────────────────────────────────
ANTHROPIC_FOUNDRY_API_KEY  = os.getenv("ANTHROPIC_FOUNDRY_API_KEY", "")
ANTHROPIC_FOUNDRY_RESOURCE = os.getenv("ANTHROPIC_FOUNDRY_RESOURCE", "")
ANTHROPIC_FOUNDRY_BASE_URL = os.getenv(
    "ANTHROPIC_FOUNDRY_BASE_URL",
    f"https://{ANTHROPIC_FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic"
    if ANTHROPIC_FOUNDRY_RESOURCE else "",
)

# Claude model deployment names (set via env or default to standard names)
SONNET_MODEL = os.getenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-4-6")
HAIKU_MODEL  = os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL",  "claude-haiku-4-5")
OPUS_MODEL   = os.getenv("ANTHROPIC_DEFAULT_OPUS_MODEL",   "claude-opus-4-6")

# Which tier to use: "sonnet" | "haiku" | "opus"  (default: sonnet)
MODEL_TIER = os.getenv("MODEL_TIER", "sonnet")

# ── OpenAI fallback (used when Foundry key is absent) ───────────────────────
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

# ── Embeddings ───────────────────────────────────────────────────────────────
# Azure OpenAI embeddings (optional — falls back to standard OpenAI)
AZURE_OPENAI_API_KEY         = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT        = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_EMBEDDING_DEPLOYMENT   = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
EMBEDDING_MODEL              = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ── Indexing ─────────────────────────────────────────────────────────────────
CHUNK_SIZE                    = 512
CHUNK_OVERLAP                 = 50
SEMANTIC_BREAKPOINT_PERCENTILE = 90

# ── Retrieval ────────────────────────────────────────────────────────────────
DENSE_K          = 10
DENSE_FETCH_K    = 30
SPARSE_K         = 10
HYBRID_WEIGHTS   = [0.35, 0.65]   # [BM25, vector] — tune per domain
RERANKER_TOP_N   = 5
FINAL_TOP_K      = 6
MULTI_QUERY_COUNT = 3

# ── Generation ───────────────────────────────────────────────────────────────
MAX_RETRIES = 2

# ── Data sources ─────────────────────────────────────────────────────────────
SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING", "")

# ── Tools ────────────────────────────────────────────────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── MCP client ───────────────────────────────────────────────────────────────
# Reuses the Foundry key if ANTHROPIC_API_KEY is not separately set
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_FOUNDRY_API_KEY
# Bearer token for Dubai Holding Foundry MCP endpoint
FOUNDRY_TOKEN = os.getenv("FOUNDRY_TOKEN", "")
MCP_SERVER_URL = os.getenv(
    "MCP_SERVER_URL",
    "https://foundry.dubaiholding.com/mcp/third-party-application/"
    "ri.third-party-applications.main.application.9b7db0c4-8264-4f17-9afc-a09b79bb54d0",
)
