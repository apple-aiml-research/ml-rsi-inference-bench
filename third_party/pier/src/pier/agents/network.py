#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from pier.models.agent.network import NetworkAllowlist


def hostname_from_url(value: Any) -> str | None:
    """Return the hostname for a URL-like config value."""
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw or raw.startswith("{env:") or raw.startswith("${"):
        return None

    has_scheme = "://" in raw
    if not has_scheme and "." not in raw and raw != "localhost" and ":" not in raw:
        return None

    parsed = urlparse(raw if has_scheme else f"https://{raw}")
    if not parsed.hostname:
        return None
    return parsed.hostname.lower().rstrip(".")


def allowlist_from_urls(
    values: Iterable[Any],
    *,
    default_domains: Iterable[str] = (),
) -> NetworkAllowlist:
    domains = {domain for domain in default_domains if domain}
    for value in values:
        if hostname := hostname_from_url(value):
            domains.add(hostname)
    return NetworkAllowlist(domains=sorted(domains))


def collect_url_values(value: Any, *, keys: set[str] | None = None) -> list[str]:
    """Collect URL-like values from nested config dictionaries.

    This intentionally keys off common provider/base-url field names instead of
    trying to interpret every string in an arbitrary config blob.
    """
    url_keys = keys or {
        "api",
        "api_base",
        "api_url",
        "base_url",
        "baseurl",
        "baseURL",
        "openai_base_url",
        "url",
    }

    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key in url_keys and isinstance(nested, str):
                found.append(nested)
            elif isinstance(nested, dict | list):
                found.extend(collect_url_values(nested, keys=url_keys))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                found.extend(collect_url_values(item, keys=url_keys))
    return found
