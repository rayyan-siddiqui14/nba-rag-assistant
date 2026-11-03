"""One interface for text generation. Swap backends via .env."""
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("LLM_BACKEND", "ollama")


def generate(prompt: str, max_tokens: int = 500) -> str:
    """Send a prompt to the configured LLM and return the text response."""
    if BACKEND == "ollama":
        import ollama
        r = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "seed": 42},
        )
        return r["message"]["content"]

    from anthropic import Anthropic
    client = Anthropic()
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text