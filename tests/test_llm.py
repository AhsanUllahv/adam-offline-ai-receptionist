from __future__ import annotations

import unittest

from assistant_app.config import AssistantConfig
from assistant_app.llm import OllamaLLM


class OllamaPromptTests(unittest.TestCase):
    def test_prompt_requires_grounded_document_answer(self) -> None:
        llm = OllamaLLM(model="test", base_url="http://localhost")

        prompt = llm._build_prompt("What is the policy?", "[manual] Policy text")

        self.assertIn("virtual receptionist", prompt)
        self.assertIn("Do not use outside knowledge", prompt)
        self.assertIn("Answer:", prompt)
        self.assertIn("Not found:", prompt)
        self.assertIn("[manual] Policy text", prompt)
        self.assertIn("robo eyes", prompt)


    def test_general_prompt_allows_receptionist_answers(self) -> None:
        llm = OllamaLLM(model="test", base_url="http://localhost")

        prompt = llm._build_general_prompt("What is VBC College?")

        self.assertIn("friendly virtual receptionist", prompt)
        self.assertIn("general questions", prompt)
        self.assertIn("Do not start with filler", prompt)
        self.assertIn("local knowledge base does not contain", prompt)

    def test_ollama_uses_generation_options_from_config(self) -> None:
        llm = OllamaLLM.from_config(
            AssistantConfig(
                ollama_model="llama3.2:1b",
                ollama_num_predict=64,
                ollama_num_ctx=1024,
                ollama_temperature=0.0,
            )
        )

        self.assertEqual(llm.model, "llama3.2:1b")
        self.assertEqual(llm.num_predict, 64)
        self.assertEqual(llm.num_ctx, 1024)
        self.assertEqual(llm.temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
