#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
# Modified by Apple Inc. in 2026.
import json
from pathlib import Path

from pier.models.trial.config import ServiceVolumeConfig

# Shared compose file paths used by both local Docker and Daytona DinD environments.
COMPOSE_DIR = Path(__file__).parent
COMPOSE_BASE_PATH = COMPOSE_DIR / "docker-compose-base.yaml"
COMPOSE_BUILD_PATH = COMPOSE_DIR / "docker-compose-build.yaml"
COMPOSE_PREBUILT_PATH = COMPOSE_DIR / "docker-compose-prebuilt.yaml"
COMPOSE_NO_NETWORK_PATH = COMPOSE_DIR / "docker-compose-no-network.yaml"
COMPOSE_WINDOWS_KEEPALIVE_PATH = COMPOSE_DIR / "docker-compose-windows-keepalive.yaml"
RESOURCES_COMPOSE_NAME = "docker-compose-resources.json"


def write_mounts_compose_file(path: Path, mounts: list[ServiceVolumeConfig]) -> Path:
    compose = {"services": {"main": {"volumes": list(mounts)}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path


def write_resources_compose_file(
    path: Path,
    *,
    cpu_request: int | None = None,
    cpu_limit: int | None = None,
    memory_request_mb: int | None = None,
    memory_limit_mb: int | None = None,
    gpu_count: int | None = None,
    shm_size_mb: int | None = None,
) -> Path:
    resources: dict[str, object] = {}
    limits: dict[str, str] = {}
    reservations: dict[str, object] = {}

    if cpu_limit is not None:
        limits["cpus"] = str(cpu_limit)
    if memory_limit_mb is not None:
        limits["memory"] = f"{memory_limit_mb}M"
    if cpu_request is not None:
        reservations["cpus"] = str(cpu_request)
    if memory_request_mb is not None:
        reservations["memory"] = f"{memory_request_mb}M"
    # Inference-Bench fork addition — local GPU container support. Request NVIDIA
    # GPUs through the Compose device-reservation spec, which `docker compose
    # up` translates into the same DeviceRequest as `docker run --gpus`. Only
    # emitted when the task asks for GPUs, so CPU-only tasks are unchanged and
    # boxes without nvidia-docker keep working for the existing (CPU) path.
    if gpu_count:
        reservations["devices"] = [
            {
                "driver": "nvidia",
                "count": gpu_count,
                "capabilities": ["gpu"],
            }
        ]

    if limits:
        resources["limits"] = limits
    if reservations:
        resources["reservations"] = reservations

    main: dict[str, object] = {"deploy": {"resources": resources}} if resources else {}
    # Inference-Bench fork addition — /dev/shm sizing. `shm_size` is a top-level
    # SERVICE key, not part of deploy.resources, and is emitted in the same
    # "<N>M" suffixed form this writer already uses for the deploy memory
    # limit/reservation (Docker parses that suffix as binary MiB). Only emitted
    # when the task declares it, so undeclared tasks keep the runtime default.
    if shm_size_mb is not None:
        main["shm_size"] = f"{shm_size_mb}M"
    compose = {"services": {"main": main}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(compose, indent=2))
    return path
