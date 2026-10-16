#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2024 OpenAI (simple-evals)
# Adapted from https://github.com/sgl-project/sglang.
# Modified by Apple Inc. in 2026.
# Adapted from https://github.com/openai/simple-evals/

"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
from typing import Optional

import pandas

from . import simple_eval_common as common
from .simple_eval_common import (
    ANSWER_PATTERN_MULTICHOICE,
    HTML_JINJA,
    Eval,
    EvalResult,
    MessageList,
    SamplerBase,
    SingleEvalResult,
    format_multichoice_question,
)


class GPQAEval(Eval):
    def __init__(
        self,
        filename: str,
        num_examples: Optional[int],
        num_threads: int,
        n_repeats: int = 1,
        seed: int = 0,
    ):
        df = pandas.read_csv(filename)
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(seed)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [
            example | {"permutation": rng.sample(range(4), 4)} for example in examples
        ]
        self.examples = examples
        self.n_repeats = n_repeats
        self.num_threads = num_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0],
                B=choices[1],
                C=choices[2],
                D=choices[3],
                Question=row["Question"],
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            if response_text is None:
                response_text = ""
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={"chars": len(response_text)},
            )

        results = common.map_with_progress(fn, self.examples, self.num_threads)
        return common.aggregate_results(results)
