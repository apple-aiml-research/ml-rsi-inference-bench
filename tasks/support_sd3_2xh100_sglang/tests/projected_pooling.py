#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
'Choose the projected CLIP output, independent of legal output naming.'
def projected_pooled(capture):
    value = capture.get("text_embeds")
    return capture.get("pooler_output") if value is None else value
