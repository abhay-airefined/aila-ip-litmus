from __future__ import annotations

import logging

from openai import AzureOpenAI, OpenAI

from app.config import settings

logger = logging.getLogger("sfas")


def _openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("SFAS_OPENAI_API_KEY is required when SFAS_LLM_PROVIDER=openai")
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(**kwargs)


def _azure_openai_client() -> AzureOpenAI:
    if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
        raise RuntimeError(
            "SFAS_AZURE_OPENAI_API_KEY and SFAS_AZURE_OPENAI_ENDPOINT are required when SFAS_LLM_PROVIDER=azure_openai"
        )
    return AzureOpenAI(
        api_key=settings.azure_openai_api_key,
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version,
    )


def _foundry_client() -> OpenAI:
    if not settings.azure_foundry_api_key or not settings.azure_foundry_base_url:
        raise RuntimeError(
            "SFAS_AZURE_FOUNDRY_API_KEY and SFAS_AZURE_FOUNDRY_BASE_URL are required when SFAS_LLM_PROVIDER=azure_foundry"
        )
    return OpenAI(api_key=settings.azure_foundry_api_key, base_url=settings.azure_foundry_base_url)


def _chat_complete(client: OpenAI | AzureOpenAI, model_name: str, prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a continuation engine for forensic attribution testing. Continue the text naturally and concisely.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=settings.llm_temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def generate_model_continuations(
    model_name: str,
    prompts: list[str],
    corpus_tokens: list[str],
    max_tokens: int = 20,
) -> list[str]:
    del corpus_tokens
    provider = settings.llm_provider.strip().lower()
    effective_max_tokens = min(max_tokens, settings.llm_max_tokens)

    if provider == "openai":
        client = _openai_client()
    elif provider == "azure_openai":
        client = _azure_openai_client()
    elif provider == "azure_foundry":
        client = _foundry_client()
    else:
        raise RuntimeError(
            f"Unsupported SFAS_LLM_PROVIDER '{settings.llm_provider}'. Use openai, azure_openai, or azure_foundry."
        )

    logger.info("model_gateway provider=%s model_name=%s prompts=%s", provider, model_name, len(prompts))
    outputs: list[str] = []
    for prompt in prompts:
        outputs.append(_chat_complete(client, model_name, prompt, effective_max_tokens))
    return outputs
