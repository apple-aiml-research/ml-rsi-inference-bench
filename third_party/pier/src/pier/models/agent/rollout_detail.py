#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from typing import Any, TypedDict


class RolloutDetail(TypedDict, total=False):
    """Details about a complete chat history.

    This represents the full conversation sequence, allowing downstream
    applications to recover the entire conversation using token IDs and logprobs.

    For agents with subagents, summarization, or other non-linear chat histories,
    multiple RolloutDetail objects can be stored in the list to represent
    different segments. By convention, the first RolloutDetail object should contain
    the main agent's conversation history.
    """

    prompt_token_ids: list[
        list[int]
    ]  # Each element contains full prompt token IDs for that turn, including previous chat history (if applicable)
    completion_token_ids: list[
        list[int]
    ]  # Each element contains response token IDs for that turn
    logprobs: list[
        list[float]
    ]  # Each element contains logprobs corresponding to completion_token_ids for that turn
    extra: dict[
        str, list[Any]
    ]  # Per-turn provider-specific data keyed by field name (e.g. {"routed_experts": [turn1_data, ...]})
