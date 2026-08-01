from math import exp
from types import SimpleNamespace

import pytest

from scaleflow.backends.vllm import (
    confidence_from_token_logprobs,
    extract_chosen_token_logprobs,
)


def test_confidence_is_geometric_mean_of_chosen_token_probabilities() -> None:
    logprobs = [-0.2, -0.6, -1.0]

    confidence = confidence_from_token_logprobs(logprobs)

    assert confidence == pytest.approx(exp(sum(logprobs) / len(logprobs)))


def test_chosen_token_logprobs_follow_generated_token_ids() -> None:
    positions = [
        {
            10: SimpleNamespace(logprob=-0.1),
            99: SimpleNamespace(logprob=-2.0),
        },
        {
            20: SimpleNamespace(logprob=-0.3),
            98: SimpleNamespace(logprob=-1.5),
        },
    ]

    assert extract_chosen_token_logprobs([10, 20], positions) == [-0.1, -0.3]


@pytest.mark.parametrize(
    ("token_ids", "positions"),
    [([], []), ([10], None), ([10], []), ([10], [{99: SimpleNamespace(logprob=-1.0)}])],
)
def test_chosen_token_logprobs_reject_missing_data(token_ids, positions) -> None:
    with pytest.raises(ValueError):
        extract_chosen_token_logprobs(token_ids, positions)
