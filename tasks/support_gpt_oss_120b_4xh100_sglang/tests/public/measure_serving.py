#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Public serving measurement with the verifier's flush-before-each-draw policy.

Use the same arguments as `python -m gpt_oss.efficiency.client measure`.
Each axis is one complete draw. A fresh prefix-cache flush precedes its warmups.
"""
import argparse
import importlib.util
import json
import secrets
import sys
import urllib.error
import urllib.request

# /public/gpt_oss contains the model-format sources. Load the measurement file
# directly so that package cannot shadow the separate /app/eval namespace.
_spec = importlib.util.spec_from_file_location(
    'infb_gpt_oss_efficiency_client', '/app/eval/gpt_oss/efficiency/client.py')
if _spec is None or _spec.loader is None:
    raise ImportError('cannot load /app/eval/gpt_oss/efficiency/client.py')
client = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = client
_spec.loader.exec_module(client)


def flush(base_url):
    entry = {'at': client.utcnow()}
    try:
        request = urllib.request.Request(base_url.rstrip('/')+'/flush_cache', data=b'', method='POST')
        with urllib.request.urlopen(request, timeout=60) as response:
            entry.update(status=response.status, body=response.read(200).decode(errors='replace'))
    except urllib.error.HTTPError as exc:
        entry['status'] = exc.code
    except Exception as exc:
        entry['status'] = f'{type(exc).__name__}: {exc}'[:200]
    # Match the verifier: preserve an unsuccessful flush as an observation;
    # the client's cached-token flags still diagnose cache reuse in the draw.
    print('INFB_CACHE_FLUSH '+json.dumps(entry), flush=True)
    return entry


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if '-h' in argv or '--help' in argv:
        return client.main(['measure','--help'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--axis', choices=(*client.AXES,'both'), default='both')
    parser.add_argument('--seed')
    parser.add_argument('--seed-file')
    args, _ = parser.parse_known_args(argv)
    if not args.seed and not args.seed_file:
        argv += ['--seed', secrets.token_hex(8)]
    axes = list(client.AXES) if args.axis=='both' else [args.axis]
    result = 0
    for axis in axes:
        flush(args.base_url)
        result = max(result, client.main(['measure',*argv,'--axis',axis]))
    return result


if __name__=='__main__':
    raise SystemExit(main())
