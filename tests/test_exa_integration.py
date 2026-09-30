import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))
ZssmExplain = importlib.import_module(f"{PROJECT_ROOT.name}.main").ZssmExplain


class _FakeService:
    def __init__(self, results):
        self.results = results
        self.wait_timeout = None
        self.search_args = None

    async def wait_ready(self, timeout):
        self.wait_timeout = timeout

    async def search(self, query, *, num_results, timeout):
        self.search_args = (query, num_results, timeout)
        return self.results


class ExaIntegrationTest(unittest.TestCase):
    @staticmethod
    def _plugin(service):
        plugin = object.__new__(ZssmExplain)
        plugin.config = {
            "exa_search_enable": True,
            "exa_search_max_results": 2,
            "exa_search_timeout_sec": 4,
        }
        star_cls = types.SimpleNamespace(get_service=lambda api_version=1: service)
        plugin.context = types.SimpleNamespace(
            get_registered_star=lambda name: types.SimpleNamespace(
                activated=True, star_cls=star_cls
            )
        )
        return plugin

    def test_search_results_are_formatted_for_llm(self):
        service = _FakeService(
            [
                {
                    "title": "AstrBot",
                    "url": "https://example.com/astrbot",
                    "highlights": ["插件框架", "支持扩展"],
                }
            ]
        )
        plugin = self._plugin(service)

        context = asyncio.run(plugin._search_exa_context("AstrBot 是什么"))

        self.assertIn("AstrBot", context)
        self.assertIn("https://example.com/astrbot", context)
        self.assertEqual(service.search_args, ("AstrBot 是什么", 2, 4))
        self.assertEqual(service.wait_timeout, 4)

    def test_missing_exa_service_falls_back_without_context(self):
        plugin = object.__new__(ZssmExplain)
        plugin.config = {"exa_search_enable": True}
        plugin.context = types.SimpleNamespace(get_registered_star=lambda name: None)

        self.assertEqual(asyncio.run(plugin._search_exa_context("测试")), "")


if __name__ == "__main__":
    unittest.main()
