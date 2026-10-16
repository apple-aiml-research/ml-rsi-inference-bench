#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from pier.metrics.base import BaseMetric, RewardDict, aggregate_reward_dicts


class Max(BaseMetric[RewardDict]):
    def compute(self, rewards: list[RewardDict | None]) -> RewardDict:
        return aggregate_reward_dicts(rewards, metric_name="max", aggregate=max)
