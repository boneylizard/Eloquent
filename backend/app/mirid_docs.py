from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse


router = APIRouter(tags=["Mirid documentation"])

MIRID_LLM_GUIDE = """# Mirid — AI-readable user guide

Mirid is a Windows desktop application for local and hosted AI chat, character roleplay, speech, transcription, image generation, documents, and OpenAI-compatible serving.

## Teach a newcomer

When explaining Mirid, begin with the user's goal. Do not ask them to configure every engine. Recommend the shortest working path, distinguish local models from paid APIs, and warn before any action that can spend money or download a large file.

## Fastest first chat

1. Open Settings → Models & APIs → Model Library.
2. For the easiest hosted start, connect OpenRouter and select its free router, or connect NanoGPT.
3. For NanoGPT Pro usage, choose Mirid's subscription catalogue filter, then select an included model. Mirid keeps the standard NanoGPT API endpoint; the account attached to the API key applies eligible subscription usage.
4. For private local inference, open Hugging Face, choose Mirid's Pick that fits the detected VRAM, and download its GGUF file.
5. Return to Chat, choose the model, then begin a conversation.

## Model sources

### NanoGPT

Mirid's recommended hosted option for frequent personal roleplay. NanoGPT currently advertises 60 million included input-token units per week and 100 included images per day on Pro. Most included text models use one allowance unit per input token; some are marked 2x. Mirid uses the standard NanoGPT API endpoint; eligible models are covered by the subscription attached to the API-key account and appear as free in NanoGPT usage. Included models and limits can change. Subscription use is personal and non-commercial. The live NanoGPT Subscription and Terms pages govern.

- Subscription: https://nano-gpt.com/subscription
- API keys: https://nano-gpt.com/api
- Add pay-as-you-go credit: https://nano-gpt.com/balance
- Live help and limits: https://nano-gpt.com/help
- Pricing: https://nano-gpt.com/pricing

### OpenRouter

Useful for trying many providers through one API key. Mirid places OpenRouter's free router first and shows published model prices where available. Free routes still require an OpenRouter key. Paid use draws from OpenRouter credit.

- API keys: https://openrouter.ai/settings/keys
- Add credit: https://openrouter.ai/settings/credits
- Models and pricing: https://openrouter.ai/models

### Hugging Face

Use Hugging Face to download local GGUF text models and supported image or speech files. Public repositories usually need no token. Gated or private repositories need a read or fine-grained token.

- Models: https://huggingface.co/models
- Access tokens: https://huggingface.co/settings/tokens

### Direct frontier APIs

OpenAI, Anthropic, Gemini, Mistral, xAI, and other provider APIs are billed separately from consumer chat subscriptions. A ChatGPT or Claude subscription does not normally fund developer API calls. Mirid reads the models available to the user's API key rather than relying on a frozen list.

## Choosing local or hosted

- Choose local GGUF when privacy, offline use, predictable cost, or custom roleplay models matter most.
- Choose NanoGPT when convenience, frequent personal roleplay, and access to a changing catalogue matter most.
- Choose OpenRouter when broad comparison, free routes, or provider choice matters most.
- Choose a direct frontier API when the user specifically needs that provider's models or account controls.

## Character roleplay

Import compatible PNG or JSON character cards from Characters, or use Character Studio to build one conversationally. A character card defines identity, scenario, greeting, examples, and optional lore; it is not the language model itself.

## Voice and transcription

Mirid can run local TTS and STT engines or selected NanoGPT audio models. Local model files load only when used. If speech appears idle, watch the playback state: first inference can take longer while an engine wakes.

## Images

Mirid's local image engine uses stable-diffusion.cpp. It can also use configured NanoGPT image models. Local generation requires a separate compatible image checkpoint; installing a text model does not install one.

If the image panel reports that no local image model was found, choose Find an image model. Hugging Face is the primary source. Civitai is optional, uses Civitai's official API, and is available only where Civitai makes its service available; Mirid does not bypass regional restrictions. Automatic Civitai downloads are limited to primary Safetensors or GGUF checkpoints that pass Civitai's scans.

Prefer a self-contained checkpoint unless the model card clearly lists every required companion file. Some newer model families need separate text encoders or a VAE.

## SillyTavern

Mirid exposes OpenAI-compatible streaming chat and compatibility endpoints for speech, transcription, and local image generation. The Mirid Bridge extension adds automatic narration and microphone transcription inside SillyTavern.

## Safety and billing

- Never paste API keys into chat messages or character cards.
- Use provider spend limits where available.
- Treat catalogue prices and subscription allowances as live information that can change.
- Large local models can consume substantial disk space, RAM, and VRAM.
- Mirid listens on localhost by default. Set a remote-access password before deliberately exposing it to a network.

## Troubleshooting order

1. Confirm Mirid says the backend is ready.
2. Confirm the selected model or endpoint still exists and is enabled.
3. For APIs, confirm the key, billing mode, credit or subscription allowance, and current model eligibility.
4. For local models, confirm the file is installed and fits available RAM/VRAM at the chosen context length.
5. Retry once after reading the exact error. Do not repeatedly retry paid requests without understanding the failure.
"""


@router.get("/docs/llms.txt", response_class=PlainTextResponse)
async def mirid_llm_guide():
    return MIRID_LLM_GUIDE


@router.get("/docs/index.json")
async def mirid_docs_index():
    return {
        "product": "Mirid",
        "human_docs_path": "/docs",
        "ai_guide_path": "/docs/llms.txt",
        "topics": [
            "first chat",
            "local GGUF models",
            "NanoGPT subscription",
            "OpenRouter",
            "Hugging Face downloads",
            "frontier APIs",
            "characters",
            "voice and transcription",
            "image generation",
            "SillyTavern",
            "billing safety",
            "troubleshooting",
        ],
    }
