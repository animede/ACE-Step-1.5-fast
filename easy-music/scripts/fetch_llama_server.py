#!/usr/bin/env python
"""フォールバック用のビルド済み llama-server(CPU 版 約 25MB / CUDA 版 約 0.6〜0.8GB)と Gemma 4 E4B の GGUF(約 4.4GB)を事前取得する。

  .venv/bin/python scripts/fetch_llama_server.py            # llama-server だけ
  .venv/bin/python scripts/fetch_llama_server.py --model    # GGUF も
  .venv/bin/python scripts/fetch_llama_server.py --cpu      # CUDA 版ではなく CPU 版(既定は nvidia-smi が見えれば CUDA 版 = LOCAL_LLM_GPU=auto)

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
parser.add_argument("--cpu", action="store_true", help="CPU 版を取る(既定は GPU があれば CUDA 版)")
parser.add_argument("--gpu", action="store_true", help="CUDA 版を取る(nvidia-smi が無くても)")
args = parser.parse_args()

gpu = False if args.cpu else (True if args.gpu else m.use_gpu())
print(f"llama-server ({'CUDA' if gpu else 'CPU'}): {m.find_binary(gpu) or m.download_binary(gpu)}")
if args.model:
    print(f"model: {m.download_model()}")
else:
    print(f"model: {m.model_path()} ({'present' if os.path.exists(m.model_path()) else 'not fetched: auto on first fallback, or --model'})")
