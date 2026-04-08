from pathlib import Path

from dotenv import load_dotenv

# Load env vars (GOOGLE_API_KEY, CROSSREF_URL, ...) before any sub-agent imports
load_dotenv(Path(__file__).resolve().parent / ".env")

from . import agent  # noqa: E402
