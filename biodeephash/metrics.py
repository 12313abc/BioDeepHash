"""Exact-match verification metrics for one protected record per identity."""

from collections import Counter

from .protection import verify


def evaluate(probes_by_identity, references):
    """Score each probe once against its own record and all other records."""
    if len(references) < 2 or probes_by_identity.keys() != references.keys():
        raise ValueError("Evaluation requires matching probe and enrollment identities (at least two).")

    true_accepts = false_accepts = genuine_attempts = 0
    consistency = 0.0
    for identity, codes in probes_by_identity.items():
        counts = Counter(codes)
        if not counts:
            raise ValueError(f"No test probes for identity {identity}.")
        genuine_attempts += len(codes)
        consistency += max(counts.values()) / len(codes)
        for code, count in counts.items():
            for candidate, reference in references.items():
                if verify(code, reference):
                    if candidate == identity:
                        true_accepts += count
                    else:
                        false_accepts += count

    rates = _verification_rates(true_accepts, false_accepts, genuine_attempts, len(references))
    return {**rates, "consistency_rate": consistency / len(references)}


def evaluate_crop_votes(probes, references, minimum_votes):
    """Score one trial per original probe image and claimed identity."""
    if len(references) < 2 or minimum_votes < 1:
        raise ValueError("Crop voting requires at least two references and one vote.")
    true_accepts = false_accepts = genuine_attempts = 0
    for codes, identity in probes:
        if identity not in references or len(codes) < minimum_votes:
            raise ValueError("Probe identity or crop count is invalid.")
        genuine_attempts += 1
        counts = Counter(codes)
        for candidate, reference in references.items():
            votes = sum(count for code, count in counts.items() if verify(code, reference))
            if votes >= minimum_votes:
                if candidate == identity:
                    true_accepts += 1
                else:
                    false_accepts += 1
    if not genuine_attempts:
        raise ValueError("No test probes for crop voting.")
    return _verification_rates(true_accepts, false_accepts, genuine_attempts, len(references))


def _verification_rates(true_accepts, false_accepts, genuine_attempts, reference_count):
    impostor_attempts = genuine_attempts * (reference_count - 1)
    return {
        "true_accepts": true_accepts,
        "false_accepts": false_accepts,
        "genuine_attempts": genuine_attempts,
        "impostor_attempts": impostor_attempts,
        "gar_percent": 100 * true_accepts / genuine_attempts,
        "far_percent": 100 * false_accepts / impostor_attempts,
    }
