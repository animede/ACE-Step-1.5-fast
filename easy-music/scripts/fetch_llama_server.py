#!/usr/bin/env python
"""フォールバック用のビルド済み llama-server(約 25MB)と Gemma 4 E4B の GGUF(約 4.4GB)を事前取得する。

  .venv/bin/python scripts/fetch_llama_server.py            # llama-server だけ
  .venv/bin/python scripts/fetch_llama_server.py --model    # GGUF も

取得しなくても、初めてフォールバックしたときにアプリが自動で取得する(services/local_llm_service.py)。
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from services import local_llm_service as m  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="store_true", help="GGUF(約 4.4GB)も取得する")
args = parser.parse_args()

print(f"llama-server: {m.find_binary() or m.download_binary()}")
if args.model:
    print(f"model: {m.download_model()}")
else:
    print(f"model: {m.model_path()} ({'present' if os.path.exists(m.model_path()) else 'not fetched: auto on first fallback, or --model'})")
