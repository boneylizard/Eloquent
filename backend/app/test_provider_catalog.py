from app.provider_catalog import normalize_provider_models


def test_normalises_gemini_generate_content_models_only():
    models = normalize_provider_models("gemini", {
        "models": [
            {
                "name": "models/gemini-chat",
                "displayName": "Gemini Chat",
                "supportedGenerationMethods": ["generateContent"],
                "inputTokenLimit": 1000000,
            },
            {
                "name": "models/text-embedding",
                "supportedGenerationMethods": ["embedContent"],
            },
        ]
    })

    assert [model["id"] for model in models] == ["gemini-chat"]
    assert models[0]["context_length"] == 1000000


def test_normalises_mistral_chat_capabilities():
    models = normalize_provider_models("mistral", {
        "data": [
            {
                "id": "mistral-chat",
                "max_context_length": 32768,
                "capabilities": {"completion_chat": True, "vision": True, "function_calling": True},
            },
            {"id": "mistral-embed", "capabilities": {"completion_chat": False}},
        ]
    })

    assert len(models) == 1
    assert models[0]["capabilities"]["vision"] is True
    assert models[0]["capabilities"]["tools"] is True

