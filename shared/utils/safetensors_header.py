"""Read the JSON header of a .safetensors file without loading any tensors.

Layout: bytes 0-7 are a little-endian u64 header length N, followed by N bytes
of JSON mapping tensor names -> {dtype, shape, data_offsets}, plus an optional
"__metadata__" key holding a flat string->string map. Training tools embed
their provenance there (ai-toolkit: full config incl. trigger_word; kohya:
ss_* keys; SAI ModelSpec: modelspec.* keys).

Standalone module: no WanGP runtime imports, so it can be unit-tested anywhere.
"""

import json
import os
import struct
from collections import Counter

# A real header is a few KB to a few MB; anything bigger is corrupt or not a
# safetensors file at all.
MAX_HEADER_BYTES = 100 * 1024 * 1024

_LORA_DOWN_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora.A.weight", ".lora.down.weight")
# Same prefixes mmgp strips before matching module names.
_STRIP_PREFIXES = ("diffusion_model.", "transformer.")


def read_safetensors_header(path):
    """Return (metadata, tensors) from a safetensors file.

    metadata is the __metadata__ map with best-effort JSON parsing per value
    (training tools serialize nested config as JSON strings); tensors is the
    raw tensor index (name -> {dtype, shape, data_offsets}).
    """
    with open(path, "rb") as reader:
        prefix = reader.read(8)
        if len(prefix) < 8:
            raise ValueError("not a safetensors file (shorter than 8 bytes)")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            raise ValueError(f"implausible safetensors header length {header_len}")
        raw = reader.read(header_len)
        if len(raw) < header_len:
            raise ValueError("truncated safetensors header")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"safetensors header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    metadata = header.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = {key: _maybe_json(value) for key, value in metadata.items()}
    return metadata, header


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _strip_prefix(key):
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def guess_lora_format(tensor_keys):
    """Heuristic key-format classifier (tuned for Krea 2 LoRAs).

    - "wangp-native": txtfusion.* / attn wq|wk|wv module names -> loads in WanGP
    - "diffusers":    text_fusion.* / attn to_q|to_k|to_v|to_gate names
                      (ai-toolkit / OneTrainer) -> does NOT load in WanGP
                      (deepbeepmeep/Wan2GP#1994)
    - "kohya":        lora_unet_* / lora_te* flat underscore names
    - "unknown"
    """
    kohya = diffusers = native = False
    for key in tensor_keys:
        stripped = _strip_prefix(key)
        if stripped.startswith(("lora_unet_", "lora_te")):
            kohya = True
        if "text_fusion." in stripped or any(marker in stripped for marker in (".to_q.", ".to_k.", ".to_v.", ".to_gate.", ".to_out")):
            diffusers = True
        if "txtfusion." in stripped or any(marker in stripped for marker in (".attn.wq.", ".attn.wk.", ".attn.wv.")):
            native = True
    if diffusers and not native:
        return "diffusers"
    if native and not diffusers:
        return "wangp-native"
    if kohya:
        return "kohya"
    return "unknown"


def summarize_tensors(tensors, top_prefixes=20):
    keys = list(tensors.keys())
    prefixes = Counter(_strip_prefix(key).split(".", 1)[0] for key in keys)
    rank_guess = None
    for key in keys:
        if key.endswith(_LORA_DOWN_SUFFIXES):
            shape = (tensors[key] or {}).get("shape") or []
            if len(shape) == 2:
                rank_guess = min(shape)
                break
    summary = {
        "tensor_count": len(keys),
        "rank_guess": rank_guess,
        "format_guess": guess_lora_format(keys),
        "key_prefixes": dict(prefixes.most_common(top_prefixes)),
    }
    if len(prefixes) > top_prefixes:
        summary["key_prefixes_omitted"] = len(prefixes) - top_prefixes
    return summary


def inspect_safetensors(path, include_tensors=False):
    """One-call header inspection: metadata + tensor summary (+ full index on request)."""
    metadata, tensors = read_safetensors_header(path)
    result = {
        "file": os.path.basename(path),
        "size_bytes": os.path.getsize(path),
        "metadata": metadata,
    }
    result.update(summarize_tensors(tensors))
    if include_tensors:
        result["tensors"] = tensors
    return result
