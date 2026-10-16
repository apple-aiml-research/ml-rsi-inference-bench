#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Public examples of the independent QSA/PLE mathematics (CPU, small tensors)."""
import torch

from qwen38.reference import ple, qsa


def test_admissible_selections_allow_near_ties_only():
    scores = torch.tensor([3.0, 2.0, 1.99, 0.5])
    assert qsa.admissible(scores, torch.tensor([0, 1]))[0]
    assert qsa.admissible(scores, torch.tensor([0, 2]))[0]        # 1.99 vs 2.0: inside the allowance
    assert not qsa.admissible(scores, torch.tensor([0, 3]))[0]    # 0.5 vs 2.0: outside it


def test_a_row_attends_to_its_groups_and_its_open_tail():
    tokens = qsa.attended_tokens(torch.tensor([1]), position=9, ratio=4)
    assert tokens.tolist() == [4, 5, 6, 7, 8, 9]                  # group 1, then the open tail 8..9
    assert qsa.visible_groups(9, 4) == 2 and qsa.visible_groups(2, 4) == 0


def test_ngram_rows_never_cross_an_end_of_sequence():
    kwargs = dict(multipliers=[1, 3, 5], sizes=[97, 89, 83, 79], offsets=[0, 97, 186, 269], heads_per_ngram=2,
                  context_len=2, eos=0)
    after_eos = ple.ngram_rows([7, 0], [9], **kwargs)
    fresh = ple.ngram_rows([0, 0], [9], **kwargs)
    assert torch.equal(after_eos, fresh)


def test_the_convolution_is_causal():
    torch.manual_seed(0)
    channels, kernel, dilation = 6, 4, 3
    weight = torch.randn(channels, kernel)
    history = torch.zeros((kernel - 1) * dilation, channels)
    gated, conv_in = torch.randn(5, channels), torch.randn(5, channels)
    out = ple.convolution_output(gated, conv_in, history, weight, dilation=dilation)
    changed = conv_in.clone()
    changed[4] += 1.0
    again = ple.convolution_output(gated, changed, history, weight, dilation=dilation)
    assert torch.equal(out[:4], again[:4]) and not torch.equal(out[4], again[4])
