"""Versioned, integrity-checked persistence helpers for the public SDK API."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

SDK_VERSION = "0.1.0"
FORMAT_VERSION = 1
_HEADER = struct.Struct(">8sHQ32s")


def pack_payload(magic: bytes, value: dict[str, Any]) -> bytes:
    if len(magic) != 8:
        raise ValueError("persistence magic must be exactly 8 bytes")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _HEADER.pack(magic, FORMAT_VERSION, len(encoded), hashlib.sha256(encoded).digest()) + encoded


def unpack_payload(payload: bytes, expected_magic: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or len(payload) < _HEADER.size:
        raise ValueError("invalid persistence payload")
    magic, version, payload_size, checksum = _HEADER.unpack(payload[: _HEADER.size])
    if magic != expected_magic:
        raise ValueError("invalid persistence magic")
    if version != FORMAT_VERSION:
        raise ValueError("unsupported persistence format version")
    encoded = payload[_HEADER.size :]
    if len(encoded) != payload_size:
        raise ValueError("invalid persistence payload length")
    if hashlib.sha256(encoded).digest() != checksum:
        raise ValueError("persistence checksum mismatch")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid persistence JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("invalid persistence document")
    if value.get("format_version") != FORMAT_VERSION:
        raise ValueError("persistence metadata format version mismatch")
    if not isinstance(value.get("sdk_version"), str):
        raise ValueError("persistence metadata is missing SDK version")
    return value
