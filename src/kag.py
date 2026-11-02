"""Run the Kaggle CLI with the token from .env, so the token never appears in a command line or log."""
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
os.environ["KAGGLE_API_TOKEN"] = dotenv_values(ROOT.parent / ".env")["KAGGLE_API_TOKEN"].strip()
os.environ["KAGGLE_USERNAME"] = "imadlem"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.exit(subprocess.call([str(Path(sys.executable).parent / "kaggle.exe"), *sys.argv[1:]]))
