#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Local GPU container support for the Docker environment (Inference-Bench fork).

Covers ``write_resources_compose_file(..., gpu_count=...)``, which emits a
Compose device reservation so ``docker compose up`` attaches NVIDIA GPUs the
same way ``docker run --gpus`` does. CPU-only tasks (gpu_count 0/None) must be
byte-for-byte unaffected.
"""

import json

from pier.environments.docker import write_resources_compose_file


def _resources(path):
    return json.loads(path.read_text())["services"]["main"]["deploy"]["resources"]


def test_gpu_count_emits_nvidia_device_reservation(tmp_path):
    path = tmp_path / "docker-compose-resources.json"
    write_resources_compose_file(path, gpu_count=2)

    devices = _resources(path)["reservations"]["devices"]
    assert devices == [
        {"driver": "nvidia", "count": 2, "capabilities": ["gpu"]}
    ]


def test_gpu_reservation_coexists_with_cpu_memory(tmp_path):
    path = tmp_path / "docker-compose-resources.json"
    write_resources_compose_file(
        path,
        cpu_request=4,
        memory_request_mb=8192,
        gpu_count=1,
    )

    reservations = _resources(path)["reservations"]
    assert reservations["cpus"] == "4"
    assert reservations["memory"] == "8192M"
    assert reservations["devices"][0]["count"] == 1


def test_no_gpu_leaves_cpu_only_compose_unchanged(tmp_path):
    """CPU-only tasks must not gain a devices reservation."""
    gpu0 = tmp_path / "gpu0.json"
    none = tmp_path / "none.json"
    write_resources_compose_file(gpu0, cpu_request=1, memory_request_mb=512, gpu_count=0)
    write_resources_compose_file(none, cpu_request=1, memory_request_mb=512)

    for path in (gpu0, none):
        reservations = _resources(path)["reservations"]
        assert "devices" not in reservations
        assert reservations == {"cpus": "1", "memory": "512M"}
