"""Easy Music の生成ルートの小さな回帰テスト(ACE サーバ不要)。

  cd easy-music && .venv/bin/python -m unittest routers.generate_test
"""
import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routers import generate as g  # noqa: E402


class GenerateRouteTests(unittest.TestCase):
    def test_sft_batch_clamp(self) -> None:
        # 2026-11-21: 定数 SFT_MAX_BATCH_SIZE が消えて、SFT を選ぶと /api/generate が NameError で 500 になっていた
        self.assertEqual(2, g.clamp_batch_for_sft("acestep-v15-xl-sft", 4))
        self.assertEqual(4, g.clamp_batch_for_sft("acestep-v15-turbo", 4))
        self.assertEqual(1, g.clamp_batch_for_sft(None, 1))

    def test_generate_music_sft_reaches_ace_client(self) -> None:
        sent = {}

        async def fake_release(**kw):
            sent.update(kw)
            return {"data": {"task_id": kw.get("task_id"), "queue_position": 1}}

        async def no_llm():
            return None

        orig_release, orig_llm = g.ace_step_client.release_task, g._release_local_llm
        g.ace_step_client.release_task, g._release_local_llm = fake_release, no_llm
        try:
            r = asyncio.run(g.generate_music(g.GenerateRequest(prompt="p", lyrics="[inst]", model="acestep-v15-xl-sft", batch_size=4)))
        finally:
            g.ace_step_client.release_task, g._release_local_llm = orig_release, orig_llm
            g._ace_finished()
        self.assertEqual(sent["batch_size"], 2)
        self.assertEqual(sent["model"], "acestep-v15-xl-sft")
        self.assertEqual(r.task_id, sent["task_id"])


if __name__ == "__main__":
    unittest.main()
