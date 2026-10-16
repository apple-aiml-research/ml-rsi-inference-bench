#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private evaluation probes: case construction, interventions and production observation.

These modules run in the separately built verifier only. They import the trusted,
restored public mathematics, adapters and measurement primitives (`gpt_oss.*` from
`/tests/public_eval`) and never ship in the agent's image.
"""
