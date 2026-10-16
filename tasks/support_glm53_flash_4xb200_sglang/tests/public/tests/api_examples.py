#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Two representative API requests at either required GLM-5.3 deployment."""
from __future__ import annotations

import argparse
import json

import requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:30000')
    args = parser.parse_args(argv)

    def post(path, body):
        response = requests.post(args.base_url.rstrip('/') + path, json=body, timeout=180)
        response.raise_for_status()
        return response.json()

    chat = post('/v1/chat/completions', {
        'model': '/model/zai-org/GLM-5.3-Flash',
        'messages': [{'role': 'user', 'content': 'Describe a public library in one sentence.'}],
        'temperature': 0, 'max_tokens': 64,
    })
    assert isinstance(chat['choices'][0]['message'], dict), 'missing assistant message'

    paragraph = ('The library lends books and provides quiet reading rooms. '
                 'Readers can search the catalogue, ask for information, and return borrowed books.\n')
    completion = post('/generate', {
        'text': paragraph * 160 + '\nSummarize the services described above:\n',
        'sampling_params': {'temperature': 0, 'max_new_tokens': 64},
        'return_logprob': True, 'logprob_start_len': 0,
    })
    assert isinstance(completion['text'], str), 'missing completion text'
    metadata = completion['meta_info']
    assert isinstance(metadata['input_token_logprobs'], list), 'missing prompt log-probabilities'
    print(json.dumps({
        'chat_message': chat['choices'][0]['message'],
        'completion': completion['text'],
        'prompt_tokens': metadata.get('prompt_tokens'),
        'prompt_logprob_entries': len(metadata['input_token_logprobs']),
        'scope': 'API examples only; numerical fidelity and task acceptance are not determined',
    }, indent=2))


if __name__ == '__main__':
    main()
