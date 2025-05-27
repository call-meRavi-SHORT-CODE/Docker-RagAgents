import os
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = "chroma_db"
PDF_PATH = "docker_cheatsheet.pdf"

os.environ.setdefault("USER_AGENT", "Doc_search_Agent")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
