from typing import Any


def response_to_text(response: Any) -> str:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Unexpected completion response shape: {type(response)!r}"
        ) from exc

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    # some backends return the text as a list of content blocks
    if isinstance(content, (list, tuple)):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()

    return str(content).strip()
