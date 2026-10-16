#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Client for the loaded native KDA layers of a running server; only standard-library imports.

Start the server with `/app/eval/kimi_linear/launch.py`. It loads the worker extension
`worker.KDAWorker` from this directory with VLLM_SERVER_DEV_MODE=1, whose RPCs this client calls on
all four TP ranks through `/collective_rpc`:

    python client.py --url http://127.0.0.1:8000 --mode inspect --output /tmp/kda-inspect.json
    python client.py --url http://127.0.0.1:8000 --mode sample  --output /tmp/kda-sample.json
    python client.py --url http://127.0.0.1:8000 --mode timing [--family N] --output /tmp/kda-timing.json

* inspect: discovery only -- every rank's native KDA layers, their source identities and state
  shapes and strides. It invokes nothing and establishes no numerical result.
* sample: the four sample cases of protocol.json on the first discovered layer, compared with the
  independent mathematics under numerical-policy.json; the report says whether the sample meets
  the policy (exit 0), fails it (exit 1) or is incomplete (exit 2).
* timing: the complete-layer timing families of protocol.json, with the numerical checks before
  and after each family; the report says whether those checks meet the policy. No speed bar is
  supplied, so timing never states an efficiency verdict.

Discovery, callable compatibility, sample correctness and consumption by ordinary serving are
different facts; none of these modes establishes the others, and none determines task acceptance.
Each invocation is bounded by the published component-invocation limit (../operation-limits.json).
"""

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
LIMITS = json.loads((HERE.parent / 'operation-limits.json').read_text())
INVOCATION_LIMIT_S = float(LIMITS['limits_s']['component_invocation'])
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


def _assessment():
    import importlib.util
    spec = importlib.util.spec_from_file_location('infb_kda_assessment', HERE / 'assessment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Client:
    def __init__(self, url, timeout=INVOCATION_LIMIT_S):
        self.url, self.timeout = url.rstrip('/'), timeout

    def post(self, route, body=None):
        data = json.dumps(body or {}).encode()
        request = Request(self.url + route, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    def rpc(self, method, **kwargs):
        result = self.post('/collective_rpc', {'method': 'infb_kda_' + method,
                           'kwargs': {k: str(v) for k, v in kwargs.items()}, 'timeout': self.timeout})
        ranks = result['results']
        if sorted(r['rank'] for r in ranks) != [0, 1, 2, 3]:
            raise AssertionError(f'expected all four TP ranks: {ranks}')
        return ranks

    def timing(self, seed=42817, draws=None, repeats=None, checkpoint=None, family=None):
        """Every family (or one), one RPC per family; the loaded implementation must not change between them."""
        draws = PROTOCOL['timing']['draws'] if draws is None else draws
        repeats = PROTOCOL['timing']['repeats'] if repeats is None else repeats
        combined = [{'rank': rank, 'families': []} for rank in range(4)]
        for index in (range(8) if family is None else [family]):
            result = self.rpc('timing', seed=seed, draws=draws, repeats=repeats, family=index)
            for rank in result:
                target = combined[rank['rank']]
                if target.get('layers', rank['layers']) != rank['layers']:
                    raise AssertionError('loaded implementation changed between timing families')
                target['layers'] = rank['layers']
                target['families'].extend(rank['families'])
            if checkpoint is not None:
                checkpoint(combined)
        return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--mode', choices=['inspect', 'sample', 'timing'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, help='sample: default protocol.json sample.seed_default; timing: 42817')
    parser.add_argument('--timeout', type=float, default=INVOCATION_LIMIT_S,
                        help='seconds one RPC may take (default: the published component-invocation limit)')
    parser.add_argument('--draws', type=int, help='timing draws per family (default: protocol.json)')
    parser.add_argument('--repeats', type=int, help='timed repetitions per draw (default: protocol.json)')
    parser.add_argument('--family', type=int, choices=range(8), help='time one published family (default: all)')
    args = parser.parse_args()
    client = Client(args.url, args.timeout)
    start = time.time()
    seed = args.seed if args.seed is not None else (
        PROTOCOL['sample']['seed_default'] if args.mode == 'sample' else 42817)
    record = {'schema': 'infb-kda-public/2', 'mode': args.mode, 'seed': seed, 'url': args.url,
              'started_unix': start, 'completed': False}
    rc = 2

    def save_partial(result):
        record['result'] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')

    try:
        if args.mode == 'inspect':
            result = client.rpc('inspect')
            record.update(result=result, completed=True)
            rc = 0
        elif args.mode == 'sample':
            result = client.rpc('sample', seed=seed)
            record.update(result=result, completed=True)
            record['assessment'] = _assessment().sample_report(result)
        else:
            result = client.timing(seed, args.draws, args.repeats, checkpoint=save_partial, family=args.family)
            record.update(result=result, completed=True)
            record['assessment'] = _assessment().timing_report(result, family=args.family)
        if 'assessment' in record:
            rc = _assessment().exit_code(record['assessment'])
            print(json.dumps({k: v for k, v in record['assessment'].items() if k != 'worst'}, indent=2))
    except Exception as exc:
        record.update(completed=False, error=repr(exc))
        raise
    finally:
        record['elapsed_s'] = time.time() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    return rc


if __name__ == '__main__':
    sys.exit(main())
