#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private build audit; bind-mount it, never add it to the agent image.

Inspect source, installed code, bytecode, archives and caches, including every
Git object rather than reachable history only. --filesystem-only also scans
individual image layers before their whiteouts are applied. Checkpoint tensor
payloads are excluded; their text/configuration files are still inspected.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
from pathlib import Path
import subprocess
import tarfile
import zipfile
import zlib

FORBIDDEN_BLOBS = {
    'c573792330aafd1ca8bf957688e4ac13908d9511',
    'd6aa06f3a3d5b0eb56a86b0cb2b907f7e12e1718',
    'd996f3977a150e5a5a9c22770a3ec811f45c0af9',
    'd76e2705a7bf2b31bf65a70ed92b3c940dda8783',
}
MARKERS = tuple(name.encode() for name in (
    '_qsa_pre_indexer_kernel', '_qsa_mqa_paged_kernel', '_expand_qsa_indices_kernel',
    '_qsa_sparse_paged_gqa_splitk_kernel', '_qsa_merge_splitk_kernel',
    '_compress_qsa_groups_kernel', '_ple_conv_writeback_kernel', '_ple_gate_kernel',
))
ROOTS = ('app', 'opt', 'public', 'root', 'tmp', 'usr/local/lib', 'var/cache', 'hf')


def inspect_stream(stream, label, hits):
    inspect_blocks(iter(lambda: stream.read(2 ** 20), b''), label, hits)


def inspect_blocks(blocks, label, hits):
    tail = b''
    for block in blocks:
        value = tail + block
        for marker in MARKERS:
            if marker in value:
                hits.append({'path': label, 'marker': marker.decode()})
        tail = value[-256:]


def inspect_gz(path, hits):
    """Decode bytes, never deserialize the joblib compatibility fixtures.

    Joblib historically used .gz for zlib streams, with or without a ZF
    length header. Treating those fixtures as gzip makes a readable image
    appear unreadable. Unknown/corrupt formats still fail the audit.
    """
    with path.open('rb') as source:
        magic = source.read(2)
        source.seek(0)
        if magic == b'\x1f\x8b':
            with gzip.GzipFile(fileobj=source) as stream:
                inspect_stream(stream, str(path), hits)
            return
        expected = None
        if magic == b'ZF':
            source.read(2)
            expected = int(source.read(len(hex(2 ** 64))), 16)
            # An older joblib version included one additional header space.
            if source.read(1) != b' ': source.seek(-1, 1)
        decoder = zlib.decompressobj()
        decoded = 0
        def blocks():
            nonlocal decoded
            while chunk := source.read(2 ** 20):
                while chunk:
                    value = decoder.decompress(chunk, 2 ** 20)
                    chunk = decoder.unconsumed_tail
                    decoded += len(value)
                    yield value
            value = decoder.flush()
            decoded += len(value)
            yield value
        inspect_blocks(blocks(), str(path), hits)
        if not decoder.eof or decoder.unused_data:
            raise ValueError('incomplete zlib stream or trailing uninspected bytes')
        if expected is not None and decoded != expected:
            raise ValueError(f'ZF length mismatch: {decoded} != {expected}')


def filesystem(root):
    hits, scanned, errors = [], 0, []
    # Hub blobs have digest-only names; identify their tensor payloads through
    # the snapshot links rather than accidentally scanning 185 GB of weights.
    tensor_blobs = {path.resolve() for path in (root / 'hf').glob('**/snapshots/*/*.safetensors')}
    for name in ROOTS:
        base = root / name
        if not base.exists(): continue
        for path in base.rglob('*'):
            if path.is_symlink() or not path.is_file() or '.git' in path.parts:
                continue
            if path.suffix == '.safetensors' or path in tensor_blobs: continue
            scanned += 1
            try:
                if path.suffix in ('.whl', '.zip'):
                    with zipfile.ZipFile(path) as archive:
                        for member in archive.infolist():
                            if not member.is_dir():
                                with archive.open(member) as file:
                                    inspect_stream(file, str(path) + '!' + member.filename, hits)
                elif any(path.name.endswith(suffix) for suffix in ('.tar', '.tar.gz', '.tgz', '.tar.xz', '.tar.bz2')):
                    with tarfile.open(path, mode='r|*') as archive:
                        for member in archive:
                            if member.isfile():
                                with archive.extractfile(member) as file:
                                    inspect_stream(file, str(path) + '!' + member.name, hits)
                elif path.suffix == '.gz':
                    inspect_gz(path, hits)
                else:
                    opener = {'.xz': lzma.open, '.bz2': bz2.open}.get(path.suffix, open)
                    with opener(path, 'rb') as file: inspect_stream(file, str(path), hits)
            except (OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile, zlib.error) as exc:
                errors.append({'path': str(path), 'error': repr(exc)})
    return {'files': scanned, 'hits': hits, 'unreadable': errors}


def git_audit(root):
    app = root / 'app'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(app), *args], text=True)
    objects = git('cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype)').splitlines()
    forbidden = [row for row in objects if row.split()[0] in FORBIDDEN_BLOBS]
    refs = git('for-each-ref', '--format=%(refname)').splitlines()
    non_ancestral = [ref for ref in refs if subprocess.run(
        ['git', '-C', str(app), 'merge-base', '--is-ancestor', ref, 'HEAD'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode]
    unreachable = git('fsck', '--full', '--no-reflogs', '--unreachable').strip()
    alternates = (app / '.git/objects/info/alternates').exists()
    return {'objects': len(objects), 'forbidden': forbidden, 'refs': refs,
            'non_ancestral': non_ancestral, 'unreachable': unreachable, 'alternates': alternates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'))
    parser.add_argument('--filesystem-only', action='store_true')
    args = parser.parse_args()
    report = filesystem(args.root)
    failed = bool(report['hits'] or report['unreadable'])
    if not args.filesystem_only:
        report['git'] = git_audit(args.root)
        failed |= any(report['git'][key] for key in ('forbidden', 'non_ancestral', 'unreachable', 'alternates'))
    report['passed'] = not failed
    print(json.dumps(report, indent=1))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
