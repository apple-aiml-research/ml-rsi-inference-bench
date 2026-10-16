#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Pier and Harbor projects
# Vendored from https://github.com/datacurve-ai/pier at 3f688a1.
set -u

reward=1

expected=$'web-search: rejected\nweb-fetch: rejected'

if [ ! -f /app/web_status.txt ]; then
  echo "missing /app/web_status.txt"
  reward=0
elif [ "$(cat /app/web_status.txt)" != "$expected" ]; then
  echo "unexpected /app/web_status.txt content:"
  cat /app/web_status.txt
  reward=0
fi

if timeout 5 bash -c '</dev/tcp/example.com/80' 2>/tmp/example-com.err; then
  echo "unexpectedly reached example.com:80"
  reward=0
else
  echo "example.com blocked as expected"
fi

echo "$reward" > /logs/verifier/reward.txt
exit 0
