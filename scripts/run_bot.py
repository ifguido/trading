#!/usr/bin/env python3
"""CryptoTrader entry point."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.engine import run


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    try:
        asyncio.run(run(config_path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
