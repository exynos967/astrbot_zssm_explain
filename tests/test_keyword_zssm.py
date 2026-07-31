import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))
ZssmExplain = importlib.import_module(f"{PROJECT_ROOT.name}.main").ZssmExplain


class FakeEvent:
    def __init__(self, head: str, text: str):
        self._chain = [{"type": "text", "data": {"text": head}}]
        self._text = text

    def get_messages(self):
        return self._chain

    def get_message_str(self):
        return self._text

    def get_self_id(self):
        return "bot"


class KeywordZssmTest(unittest.TestCase):
    @staticmethod
    def _make_plugin(trigger_text: str | None = None):
        plugin = object.__new__(ZssmExplain)
        calls = []

        plugin._is_group_allowed = lambda event: True

        def is_trigger(text: str, *, is_command: bool = False):
            calls.append((text, is_command))
            return text == trigger_text

        async def zssm(self, event):
            yield "triggered"

        plugin._is_zssm_trigger = is_trigger
        plugin.zssm = types.MethodType(zssm, plugin)
        return plugin, calls

    @staticmethod
    def _collect(plugin, event):
        async def collect():
            return [result async for result in plugin.keyword_zssm(event)]

        return asyncio.run(collect())

    def test_same_head_and_text_is_checked_once(self):
        plugin, calls = self._make_plugin()

        results = self._collect(plugin, FakeEvent("ordinary text", "ordinary text"))

        self.assertEqual(results, [])
        self.assertEqual(calls, [("ordinary text", False)])

    def test_different_text_still_uses_fallback(self):
        plugin, calls = self._make_plugin(trigger_text="zssm fallback")

        results = self._collect(plugin, FakeEvent("ordinary text", "zssm fallback"))

        self.assertEqual(results, ["triggered"])
        self.assertEqual(
            calls,
            [("ordinary text", False), ("zssm fallback", False)],
        )


if __name__ == "__main__":
    unittest.main()
