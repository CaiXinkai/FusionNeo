import torch

from fusionneo.masking import cosine_rate, junction_weights


def test_cosine_rate_endpoints():
    assert cosine_rate(0, 10, 0.15, 0.40) == 0.15
    assert abs(cosine_rate(9, 10, 0.15, 0.40) - 0.40) < 1e-9


def test_junction_has_highest_weight():
    weights = junction_weights(21, 10, boost=4.0, tau=2.0)
    assert torch.argmax(weights).item() == 10
    assert weights[10] > weights[0]

