import asyncio

from .mirid_docs import mirid_docs_index, mirid_llm_guide


def test_ai_guide_covers_provider_choice_and_billing_safety():
    guide = asyncio.run(mirid_llm_guide())
    assert "NanoGPT" in guide
    assert "OpenRouter" in guide
    assert "Hugging Face" in guide
    assert "Never paste API keys into chat messages" in guide
    assert "https://nano-gpt.com/subscription" in guide


def test_docs_index_points_to_human_and_ai_guides():
    index = asyncio.run(mirid_docs_index())
    assert index["human_docs_path"] == "/docs"
    assert index["ai_guide_path"] == "/docs/llms.txt"
