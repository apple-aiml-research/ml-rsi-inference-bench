#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""The verifier's KDA clients: the private correctness and binding invocations, and the family timing loop.

Standard-library imports only. The RPC transport is the trusted copy of the supplied `client.py`
(`Client.post`, `Client.rpc`, `Client.timing`); this module adds what only the grader runs: the private
correctness RPC, the binding workload and interventions against the separate instrumented server, and
the per-family measurement loop with its host-canary brackets. Moved from the evaluation tree's
own `kda/client.py` with unchanged workload, intervention and loop semantics.

    python orchestration.py --mode correctness --url URL --seed S --output FILE
    python orchestration.py --mode binding --url URL --output FILE [--trace-dir DIR] [--no-profile]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
from threading import Event
import time
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
TRUSTED = HERE.parent / 'eval' / 'kimi_linear' / 'kda'


def _trusted(name):
    spec = importlib.util.spec_from_file_location('infb_kda_supplied_' + name, TRUSTED / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUPPLIED = _trusted('client')
INVOCATION_LIMIT_S = SUPPLIED.INVOCATION_LIMIT_S

#: The timing families in protocol order: four prefill families, then four decode batches.
PREFILL_FAMILIES = range(4)
DECODE_FAMILIES = range(4, 8)


class Client(SUPPLIED.Client):
    def completion_body(self, tag, tokens, prompt_tokens):
        # Token IDs avoid timing a tokenizer and stay inside the real checkpoint's vocabulary.
        prompt = [100 + (i * 71 + tag * 173) % 30000 for i in range(prompt_tokens)]
        return {
            'model': 'moonshotai/Kimi-Linear-48B-A3B-Instruct', 'prompt': prompt,
            'max_tokens': tokens, 'temperature': 0, 'seed': 0, 'ignore_eos': True,
            'logprobs': 5, 'return_tokens_as_token_ids': True,
        }

    def completion(self, tag, tokens=32, prompt_tokens=128):
        return self.post('/v1/completions', self.completion_body(tag, tokens, prompt_tokens))

    def streaming_completion(self, tag, started, tokens=64, prompt_tokens=128):
        """Signal actual decode progress before the mixed workload injects new prefills."""
        body = {**self.completion_body(tag, tokens, prompt_tokens), 'stream': True}
        request = Request(self.url + '/v1/completions', data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        count, finish = 0, None
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for raw in response:
                    if not raw.startswith(b'data: '):
                        continue
                    data = raw[6:].strip()
                    if data == b'[DONE]':
                        break
                    for choice in json.loads(data).get('choices', []):
                        count += len((choice.get('logprobs') or {}).get('tokens', []))
                        if count:
                            started.set()
                        finish = choice.get('finish_reason') or finish
            if count != tokens or finish != 'length':
                raise AssertionError(f'incomplete streamed workload: {count}/{tokens}, {finish}')
            return {'tag': tag, 'prompt_tokens': prompt_tokens, 'generated_tokens': count, 'finish_reason': finish}
        except BaseException:
            started.set()  # let the coordinator observe this future's error promptly
            raise

    def binding(self, profiler=True, checkpoint=None):
        record = {'baseline': [], 'interventions': [], 'workloads': []}

        def save():
            if checkpoint is not None:
                checkpoint(record)

        save()
        self.rpc('binding', command='control', control='{}')
        if profiler:
            self.post('/start_profile')
        try:
            # Deliberate pure prefill and captured decode, then overlapping new prefills.
            record['workloads'].append(self.completion(801, 16, 512))
            save()
            with ThreadPoolExecutor(max_workers=12) as pool:
                started = [Event() for _ in range(8)]
                pending = [pool.submit(self.streaming_completion, 810 + i, started[i]) for i in range(8)]
                for signal in started:
                    if not signal.wait(self.timeout):
                        raise TimeoutError('ongoing decode did not begin before mixed prefills')
                for job in pending:
                    if job.done():
                        job.result()  # propagate a failed stream before sending more work
                pending += [pool.submit(self.completion, 830 + i, 8, 2048) for i in range(3)]
                record['workloads'].extend(job.result() for job in pending)
            record['execution'] = self.rpc('binding')
            save()
        finally:
            if profiler:
                self.post('/stop_profile')
        # Matched baseline repetitions expose natural serving noise before interventions.
        for _ in range(2):
            self.rpc('binding', command='control', control='{}')
            record['baseline'].append(self.completion(901, 16, 129))
            save()
        # Every mapped layer, across every rank. The state intervention is applied AFTER
        # the native entry returns: it cannot change the first token, only continuation.
        for layer in range(20):
            for kind, control in [('output', {'output_scale': 0.0}), ('state', {'state_add': 8.0})]:
                try:
                    self.rpc('binding', command='control', control=json.dumps({'layer': layer, **control}))
                    value = self.completion(901, 16, 129)
                    evidence = self.rpc('binding')
                    record['interventions'].append({'layer': layer, 'kind': kind,
                                                    'completion': value, 'ranks': evidence})
                    save()
                finally:
                    self.rpc('binding', command='control', control='{}')
        return record


def measure_families(axis, sample, reading, compare, *, baseline=None, own_baseline=False, after_family=None,
                     write=None, side='candidate'):
    """Time one axis family by family, each bracketed by host canary readings.

    `sample(family, attempt)` measures one family on the loaded server and returns its record
    (with `result`, one entry per TP rank); `reading()` takes one host canary reading; and
    `compare(baseline, reading)` judges it (`valid` is True, False or None). A family whose
    bracket is invalid is measured once more; a second invalid bracket marks it
    environment-invalid, which asks for a rerun and never decides a speed verdict. `baseline`
    is the reading every bracket is compared with: the trial's first whole-model reading, or with
    `own_baseline=True` the first reading taken here. `after_family(family, evidence)` may return
    True to stop after a fully measured family. `write(evidence)` persists the record after every step.
    """
    persist = write or (lambda evidence: None)
    evidence = {'schema': 'infb-kda-timing/1', 'side': side, 'axis': axis,
                'completed': False, 'result': [{'rank': i, 'families': []} for i in range(4)],
                'canary_baseline': baseline, 'canary_readings': [], 'measurements': [],
                'environment_invalid': [], 'family_assessments': []}
    started = time.monotonic()

    def take():
        value = reading()
        evidence['canary_readings'].append(value)
        if own_baseline and evidence['canary_baseline'] is None:
            evidence['canary_baseline'] = value
        return len(evidence['canary_readings']) - 1

    before = None
    try:
        for family in (PREFILL_FAMILIES if axis == 'prefill' else DECODE_FAMILIES):
            for attempt in range(2):
                if before is None:
                    before = take()
                result = sample(family, attempt)
                after = take()
                comparisons = [compare(evidence['canary_baseline'], evidence['canary_readings'][i])
                               for i in (before, after)]
                invalid = any(row['valid'] is False for row in comparisons)
                evidence['measurements'].append({'family': family, 'attempt': attempt,
                    'path': result.get('path'), 'before': before, 'after': after,
                    'canary_comparisons': comparisons, 'environment_invalid': invalid})
                before = None if invalid else after
                persist(evidence)
                if not invalid:
                    break
                if attempt == 1:
                    evidence['environment_invalid'].append(family)
            for row in result['result']:
                target = evidence['result'][row['rank']]
                if target.get('layers', row['layers']) != row['layers']:
                    raise RuntimeError('loaded implementation changed between timing families')
                target['layers'] = row['layers']
                target['families'].extend(row['families'])
            persist(evidence)
            if after_family is not None and after_family(family, evidence):
                evidence['stopped_after_failure'] = family
                persist(evidence)
                break
        evidence['completed'] = True
    finally:
        evidence['elapsed_s'] = time.monotonic() - started
        persist(evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--url', required=True)
    parser.add_argument('--mode', choices=['correctness', 'binding'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--timeout', type=float, default=INVOCATION_LIMIT_S)
    parser.add_argument('--no-profile', action='store_true')
    parser.add_argument('--trace-dir', type=Path, help='binding server profiler directory, visible to this client')
    args = parser.parse_args()
    client = Client(args.url, args.timeout)
    start = time.time()
    record = {'schema': 'infb-kda-private/1', 'mode': args.mode, 'seed': args.seed,
              'url': args.url, 'started_unix': start}

    def save_partial(result):
        record['result'] = result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**record, 'completed': False}, indent=2) + '\n')

    try:
        if args.mode == 'binding':
            result = client.binding(not args.no_profile, checkpoint=save_partial)
            if args.trace_dir is not None:
                from traces import inspect_traces
                result['traces'] = inspect_traces(args.trace_dir)
        else:
            result = client.rpc('correctness', seed=args.seed)
        record.update(result=result, completed=True)
    except Exception as exc:
        record.update(completed=False, error=repr(exc))
        raise
    finally:
        record['elapsed_s'] = time.time() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
