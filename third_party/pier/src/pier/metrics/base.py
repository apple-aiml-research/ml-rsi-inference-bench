#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")
NumericReward = float | int
RewardDict = dict[str, NumericReward]


class BaseMetric(ABC, Generic[T]):
    @abstractmethod
    def compute(self, rewards: list[T | None]) -> dict[str, float | int]:
        pass


def aggregate_reward_dicts(
    rewards: list[RewardDict | None],
    metric_name: str,
    aggregate: Callable[[list[NumericReward]], NumericReward],
) -> RewardDict:
    reward_keys = sorted(
        {key for reward in rewards if reward is not None for key in reward}
    )

    if len(reward_keys) <= 1:
        values = [
            0 if reward is None else next(iter(reward.values()), 0)
            for reward in rewards
        ]
        return {metric_name: aggregate(values)}

    return {
        key: aggregate(
            [0 if reward is None else reward.get(key, 0) for reward in rewards]
        )
        for key in reward_keys
    }
