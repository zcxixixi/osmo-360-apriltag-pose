import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional CUDA training dependency")

from tools.train_small_vla_demo import (
    SmallVlaPolicy,
    action_matrix,
    apply_actions,
    encode_instruction,
)


def test_instruction_encoder_preserves_utf8_bytes_with_padding():
    tokens = encode_instruction("夹爪", max_bytes=16)

    assert tokens.shape == (16,)
    assert np.count_nonzero(tokens) == len("夹爪".encode("utf-8"))
    assert np.all(tokens[: np.count_nonzero(tokens)] > 0)


def test_action_representation_round_trips_next_state():
    current = np.zeros((2, 14), dtype=np.float32)
    target = current.copy()
    target[:, 0:3] = [[0.01, -0.02, 0.03], [0.02, 0.01, -0.01]]
    target[:, 3:6] = [[0.1, -0.2, 0.3], [-0.1, 0.05, 0.2]]
    target[:, 6] = [0.02, 0.03]
    target[:, 7:10] = [[-0.01, 0.02, 0.01], [0.03, -0.02, 0.01]]
    target[:, 10:13] = [[-0.2, 0.1, 0.05], [0.2, -0.1, 0.03]]
    target[:, 13] = [0.01, 0.04]

    recovered = apply_actions(current, action_matrix(current, target))

    np.testing.assert_allclose(recovered, target, atol=1e-6)


def test_small_vla_policy_outputs_one_action_per_observation():
    model = SmallVlaPolicy(state_dim=14, action_dim=14)
    images = torch.zeros((3, 3, 56, 56))
    instruction = torch.zeros((3, 128), dtype=torch.long)
    state = torch.zeros((3, 14))

    output = model(images, images, instruction, state)

    assert output.shape == (3, 14)
    assert torch.isfinite(output).all()
