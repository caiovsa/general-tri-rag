import unittest
from unittest.mock import patch

from shared.config import llm_route, settings
from shared.llm import openrouter_extra_body


class LLMRouteTests(unittest.TestCase):
    def test_openrouter_route(self):
        with patch.object(settings, "OPENROUTER_API_KEY", "or-key"), \
             patch.object(settings, "OPENROUTER_API_BASE", "https://openrouter.test/api/v1"):
            route = llm_route("openrouter/qwen/qwen3.5-9b")
        self.assertEqual(route["api_key"], "or-key")
        self.assertEqual(route["api_base"], "https://openrouter.test/api/v1")

    def test_openrouter_requires_key(self):
        with patch.object(settings, "OPENROUTER_API_KEY", ""):
            with self.assertRaises(RuntimeError):
                llm_route("openrouter/qwen/qwen3.5-9b")

    def test_default_provider_route_unchanged(self):
        with patch.object(settings, "MARITACA_API_KEY", ""), \
             patch.object(settings, "MARITACA_API_BASE", ""):
            route = llm_route("gpt-6-luna")
        self.assertIsNone(route["api_key"])
        self.assertIsNone(route["api_base"])

    def test_maritaca_route_unchanged(self):
        with patch.object(settings, "MARITACA_API_KEY", "m-key"), \
             patch.object(settings, "MARITACA_API_BASE", "https://maritaca.test"):
            route = llm_route("gpt-6-luna")
        self.assertEqual(route["api_key"], "m-key")
        self.assertEqual(route["api_base"], "https://maritaca.test")


    def test_openrouter_provider_pin(self):
        with patch.object(settings, "OPENROUTER_PROVIDER_ONLY", "primeintellect/bf16"), \
             patch.object(settings, "OPENROUTER_ALLOW_FALLBACKS", False):
            body = openrouter_extra_body("none")
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertEqual(body["provider"], {
            "only": ["primeintellect/bf16"], "allow_fallbacks": False,
        })

    def test_openrouter_no_provider_pin(self):
        with patch.object(settings, "OPENROUTER_PROVIDER_ONLY", ""):
            body = openrouter_extra_body("none")
        self.assertNotIn("provider", body)


if __name__ == "__main__":
    unittest.main()
