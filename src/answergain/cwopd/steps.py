"""Step spans for sampled token sequences.

Offline data (answergain.data.prepare) tokenizes a known text, so character offsets come from the
tokenizer. A rollout is the other way round: the sampled ids are fixed and re-encoding the decoded
text may give different ids, so offsets are rebuilt from the ids themselves.
"""

import codecs
import json
import re
from functools import lru_cache

from answergain.segment.sentence import step_token_spans

_BYTE_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode, used by every byte-level BPE (Qwen, Llama-3, GPT-2)."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    codes = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codes.append(256 + extra)
            extra += 1
    return {chr(code): byte for byte, code in zip(printable, codes)}


def is_byte_level(tokenizer) -> bool:
    """Byte-level BPE (Qwen, Llama-3) vs SentencePiece-style pieces with byte fallback (Gemma)."""
    # Serializing the tokenizer costs ~0.2 s (a 150K-token vocabulary) and this runs once per rollout,
    # so the answer is kept on the tokenizer.
    cached = getattr(tokenizer, "_cwopd_byte_level", None)
    if cached is not None:
        return cached
    backend = getattr(tokenizer, "backend_tokenizer", None)
    byte_level = backend is not None and "ByteLevel" in json.dumps(json.loads(backend.to_str()).get("decoder"))
    try:
        tokenizer._cwopd_byte_level = byte_level
    except AttributeError:
        pass
    return byte_level


def token_bytes(tokenizer, token_ids: list[int]) -> list[bytes]:
    """UTF-8 bytes of each token; special tokens (EOS, <|im_end|>, <end_of_turn>) contribute no text."""
    byte_level = is_byte_level(tokenizer)
    decoder = _byte_decoder()
    special = set(tokenizer.all_special_ids)
    pieces = []
    for token_id, token in zip(token_ids, tokenizer.convert_ids_to_tokens(token_ids)):
        if token_id in special or token is None:
            pieces.append(b"")
        elif byte_level and all(char in decoder for char in token):
            pieces.append(bytes(decoder[char] for char in token))
        elif not byte_level and (match := _BYTE_FALLBACK.match(token)):
            pieces.append(bytes([int(match.group(1), 16)]))
        elif not byte_level:
            pieces.append(token.replace("▁", " ").encode("utf-8"))
        else:  # an added token outside the byte alphabet
            pieces.append(tokenizer.decode([token_id]).encode("utf-8"))
    return pieces


def decode_with_offsets(tokenizer, token_ids: list[int]) -> tuple[str, list[int]]:
    """Text of `token_ids` and the character offset at which each token starts.

    An incremental UTF-8 decoder keeps offsets monotone when a character's bytes are split across
    tokens: such a token starts where its first completed character lands.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    parts, starts, length = [], [], 0
    for piece in token_bytes(tokenizer, token_ids):
        starts.append(length)
        text = decoder.decode(piece)
        parts.append(text)
        length += len(text)
    tail = decoder.decode(b"", final=True)
    parts.append(tail)
    return "".join(parts), starts


def rollout_step_spans(tokenizer, response_ids: list[int], segmenter: str = "sentence") -> list[tuple[int, int]]:
    """Response-relative (start, end) token spans of the steps of a sampled response.

    Spans tile the whole response, so every token (including a trailing EOS) belongs to one step.
    A response without text (only special tokens) is one step.
    """
    if not response_ids:
        return []
    text, starts = decode_with_offsets(tokenizer, response_ids)
    spans = step_token_spans(text, starts, offset=0, segmenter=segmenter) if text else []
    if not spans:
        return [(0, len(response_ids))]
    # step_token_spans drops zero-token steps and starts at the first step's first token; make the
    # first span start at 0 so leading zero-width tokens are not orphaned.
    spans[0] = (0, spans[0][1])
    return spans


def step_texts(tokenizer, response_ids: list[int], spans: list[tuple[int, int]]) -> list[str]:
    return [tokenizer.decode(response_ids[start:end], skip_special_tokens=False) for start, end in spans]
