#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The verifier's private KDA package: never shipped to the agent.

It holds the correctness schedule, the GDN preservation cases, the binding instrumentation and its workload,
the family timing loop and the grader's verdicts. Everything it computes is composed from the trusted copy of the
supplied kda/ primitives at /tests/eval/kimi_linear/kda (the same bytes as /app/eval/kimi_linear/kda); it adds no
second implementation of the mathematics. `worker.py` is the server's worker extension, installed through the
evaluation tree's launch seam by the private serve wrapper and the reference launch.
"""
