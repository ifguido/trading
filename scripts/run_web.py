#!/usr/bin/env python3
"""Start the CryptoTrader Web UI."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import uvicorn

from src.web.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "scripts.run_web:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
