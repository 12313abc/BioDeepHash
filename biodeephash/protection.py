"""Cancelable templates as described in Sections III-C and III-D."""

from dataclasses import dataclass
from hashlib import sha3_512
from hmac import compare_digest
from secrets import token_bytes


@dataclass(frozen=True)
class ProtectedTemplate:
    randomizer: bytes
    digest: bytes


def protect(code: bytes, randomizer: bytes) -> bytes:
    """Hash (code XOR I1) || I2 using MSB-first, byte-aligned codes."""
    size = len(code)
    if not size or len(randomizer) != 2 * size:
        raise ValueError("The public randomizer must be twice the nonempty code length.")
    masked = bytes(bit ^ mask for bit, mask in zip(code, randomizer[:size]))
    return sha3_512(masked + randomizer[size:]).digest()


def enroll(code: bytes) -> ProtectedTemplate:
    """Issue a template with a fresh public randomizer."""
    randomizer = token_bytes(2 * len(code))
    return ProtectedTemplate(randomizer, protect(code, randomizer))


def verify(code: bytes, reference: ProtectedTemplate) -> bool:
    """Verify a probe against the protected template of a claimed identity."""
    return compare_digest(protect(code, reference.randomizer), reference.digest)
