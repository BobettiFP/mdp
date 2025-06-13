import tiktoken

ENC = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    return len(ENC.encode(text))
