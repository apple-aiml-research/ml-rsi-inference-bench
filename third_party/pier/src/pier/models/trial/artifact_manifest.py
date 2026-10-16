#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from typing import Any, Literal

from pydantic import BaseModel, Field


class ArtifactManifestEntry(BaseModel):
    source: str
    destination: str
    type: Literal["file", "directory"]
    status: Literal["ok", "failed", "empty"]


class ArtifactManifest(BaseModel):
    entries: list[ArtifactManifestEntry] = Field(default_factory=list)

    def to_json_data(self) -> list[dict[str, Any]]:
        return [entry.model_dump(mode="json") for entry in self.entries]
