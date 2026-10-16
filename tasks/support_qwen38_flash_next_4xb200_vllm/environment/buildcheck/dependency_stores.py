#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""The dependency store audit. Private: bind-mount it, never add it to the agent image.

Every Git object store on the image's filesystem is read object by object: /app's own repository, every
FetchContent or vendored store under /app/.deps (submodule stores included), Cargo's git databases and checkouts,
and any other directory that holds Git objects (an `objects` directory with pack/ or loose-object
subdirectories, whether or not a Git directory owns it). The build fails if a store

  * holds an object whose id is a withheld file's blob (WITHHELD_BLOBS: the reference solution's post-images and
    the private verifier's own files; blobs the agent's own checkout carries at HEAD, such as the empty file,
    are not withheld);
  * holds an object whose content carries a distinctive symbol (SYMBOLS: identifiers the reference solution
    defines that neither the agent's checkout nor /app's own history carries);
  * has a remote-tracking ref, alternates, or an object that its refs, HEAD and index do not reach;
  * is a dependency store (any store but /app's) holding more than its tip: a ref not at the tip, or a commit
    other than the tip. The tip is HEAD's commit, or for a bare Cargo database the locked revisions under
    refs/commit/ (reduce_git_stores.py).

It prints counts only.

    dependency_stores.py [--root /]
"""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import subprocess
import sys

WITHHELD_BLOBS = frozenset("""
    064339dc53f103899294133cfed23d38178c54b7 087f03fdcf5f325c51bba3b0a987c8a30407d8b9
    08aa05befdba2c42f9b454f73517d77bd3f693cc 0a5e471e9fd2ce0669d34962f1fec58f3455072a
    0a89d2a0356f8ef350b6b205e4f32333b3d44efd 0be75159dabd1cc87fa9aecbd7273055e98ce9ae
    0c79a1b6ef9b8f5431d58c3c46f55c3f31753d33 0e65956e968b93916faabe37bcf698ac80ee89d1
    0ee05328c9c54997eb54a34864a02d3b83a89636 1007c48a21517a4ae1678562769d78bf86e2b521
    10ed7d05fc8f5e0d10e2e6481b3b5f603cee4bef a5aca8fa09c7423a5b351fbd518819ff394a7d80
    134ab6e23878f14e760f17897a1c39b5b81ffed4 163f6786936654e841801d830c360f2100a62252
    19b35b8253cf17f07db1e21663848206ee956315 1a56194f83d4bd7323c739d5e21a1614134f811a
    1b2c1194e2a484523bdc6834ed53d1c60f6f6852 1b78b896a42f5e360bf0392c2ecf398e58d6217f
    1cbbbb45fb0991b0264a50f306377ff1fd33a860 1e6c308718f131ab63ccd8f5f91f29d3f2f222f8
    1faf89b020a32ef4b34d48ba739ab2bd9b19a338 204bccba3810bb2a093ef90e4ea4d6828f442d9c
    20c7f4e2fe7247dbd3bde3c330a4121efb62472f 22b73c936eceb2a19670ecba87ad3539e200c53c
    2404e44eb83d692ba8c8cb52955c117c93621140 257b0e7792ff9a9b28eeb62925d5f29f82629550
    258d25f271a968653f7755a73a7d607f233d2460 263c8a40352162ea4c0a38d1f05704223c6c3ec8
    27331516cbd3fa942332a7ff862a95eeb9bf4f8f 276eb31b17598d1d32f44141ffd1df50bcf1dbbf
    283e5f1523c251b1130c7aba25c464e39e98306b 291cbbb3123b0e4225389d3a623d782770f89e81
    2b16a2e4947ce20e0db832faa5400e038c6e518f 2e45db711d85e51c7779f015397f3dae19ff92b5
    33ad13499ad0439a58da355086eedff9f5215b34 9b4e9aca84d9a5f659f15807035df491f233c3ed
    37241c8a01bac1d1acc32baf423d650309e94fb6 377c6bafe841a0d385587960bd55fdf0dab6a06b
    3a355bb62e2575a35269166c188870ef6ac95927 3d1345397f0ac38189eed03248fed4e7ed45ba15
    3d5adb9014de64481efd321050df8b60a426ddeb 289da332fe35f03c64a136d4cf683d3bb112dcfc
    40f4a27d280e301ea000d042c544031d904756a2 45d8e8532e0d83abd542a6fed4eb062db2b3bb86
    46744a770647bb1c7e0992a3d7298073a21807cb 490802c3238e97bd18ebeab34b59e36b269484aa
    4af4e55732323449ba35ab40e9fa1dd381a9e222 4b4e10df581f1d1a832b10bb48fa95925acbaaf1
    4d8c1e40a5324c3f52693c30d87ecf1c5e7663e6 4dc017fade4a15b7c9dfaf2cde3bfa457b1e32ac
    4f3587841d4c45c60f272266315cbc138982227d 5112350ecd7f6f415cf792cdd57dcc772167c8df
    52067bc2fa6f6de573783a2af1ee461f3753337d 5238aab5f985abb80b0fd4c26d850bdc3299573e
    5249de7f8c11dfbedb8e04672d412f2517e4a630 53e46a9cf90f19c8a5714db1dc39671d62bcef00
    561457157d447a466bed6119ed6dcf2ceee8bd5f 5669f0f1fc3564fec46a62f255e77ff6bd40a36b
    56c33274ef9d4cd2d4f48b5a19cde57179a089bc c3327cefcfac93f667206bb4058f3198be5bacd5
    57885c9cb9db944bd32a73cc22c5f26ac354ee2d 57de5c00cdd6f11d24987e49d410db0fba510e74
    590ab3eb8d81702561fd1c6f236320cac1a807e5 59562ed8be993ad0ac3ea7df7a35585893f4fea8
    5a77448eca377f9596e25f3c697735bf0e2c1100 5b98b20564aac42e81ed45241f3c89e9c0c60842
    5d9b69529bce9c2c1545df0c12d7270347ec2adc 5f8bde0479b5cc8ba48c8fd4325b46f4ff65828a
    6038a7f0f35c4da053588cb9a0024b25ef7af53f 9dab7f334aaa35126171abc9ed60959409263963
    63615558e202209b20bc7ef44bc5c103b63d8b42 64f6b835415da4f2f706fbf2a7c5b171ee4d5d05
    6529f06e1299cfdeabfe502ca3dfc4085ce439b1 671cacd3b4626b55059d0a375f07db4198fa9b4c
    67728a2da8aa39c45e66df112e30107db8c1e13c 69215b1aa44c26f34f4139c26fafaa3cb90de8b2
    69e49d24b463f4e8056164ab5f3351b7f84c6047 6b7d8f9f9380e7fa1149792a19aea7131ab099e8
    6c15a98274294d29c3c160f72c497b76739126d0 6c196a258e19f56a7bd85166d2156a6b1a237a31
    6c40bdfceb2b5322961f82c98bdbc4243e4d86d3 6d8b8cbdbca5977fb96d4731ed7896ed415c12aa
    6dd766c966e543240ee6275624c4cf0af73082bb 821717dd63896a4684af3b32c48462d4e84fe644
    72c437c4127c7537bef068d1dc34f797562a857b 767301b36ec9dd88415b5b9f498aacf1fcf707b9
    76b7fd8b6d9012f708af9dead06454be5937ddb8 7b730523fae9a4c3190e9de14a8f7ab9811540d4
    7bccc373dde8753f9f7ad9e22648abe075144b41 7d74bd987632ddb227e407a4b92fd6a8a19afddf
    7f91f11d7c33f3f23d43d8b8951ef99353529198 38e89946e01c3ed8dee6abbc1964c446cd0c2c8b
    840d8102b989e725a0b0276b20f35f369654c0ad 8427833b121acdef5cf6da79d0c1e49a23d79fd8
    8505172da387d3f3c680f499d113c04f61b6ac5e 86b6bf0042c615a3336825a3a2cc64262c7d12c5
    88e839e63836e4f80d883663b5bdca73b087bd55 898c3a66ddcde107e5e1a69840292966f5e84c3f
    8a8f5f37221a90efae2fae3f2319a572eb38efdd 8b11f94dafb590a6bcc73ef4cb8fa258c68e1d70
    8b4d367e87057d1e745ed61ec4b5df0ce96fec14 8c12f18323e25b794ecb038ca7fa5e4549257131
    8ca503d214c8edb18ee0ac5b685d4fe23dd03e1b 8eb87e12b511d9ef6555aff0be13cb4576fb1dcf
    8f7268b0b4c840ac766f1577b74cb6d5d6bd9695 927ad50d5c961ac92d0d785833c09c49cb5619cf
    94060d1724a8c4da29c89b12ece832a065ef02fc 940b440849ac158fcc92c50d1d99cbc07389f351
    95c814e9fca081b70732312f49835004351e0544 9697aeed7f764e183e21192a6f5612fb95c1ffe3
    2b3994ffb2ce2ed40805c06ef1f491f9dc84f97c 99059c6b540c9becf85f3abccbf48e120dbf1d4f
    991c73af00c5765af1ee7844ae6d639c8ee768bc 9989c4d2a061ecfccd1a907f4f290661b8ff964b
    9ba438980cef80015a529b80c7b23842b86ba125 117c0898c6af5c584d38a3ea4d57aebb2190efe6
    9ca04c5c949ef66316bf2a52ee1c674c1d4e0f8b 9d62dfa563fee57200f26b86d3e33039597814f3
    9e533e28b483aaff4591a7cacd582c748260d1ee 9f45df115ed2cdd915a359ff7c38a8f6772af567
    9f982644b5691f8bccde0aa5229c404d394882f9 a162c39a869512d15d64139d8fe122d5558ed24d
    a1f62cd9e8f1c2781c663b24337c7e40af3f5bf3 a3357d31fb499fb56c7b065a89217bef9bc9c35a
    a4ae2bf5fbf6bf1208d8328198278f74a9a7a47f a4b4127c0b7ea55212af289672ee7c3ec3599dca
    b10694406f4028162b60c05b7c334bc77b05563b 9cd57fe2094db4aa4848f78ce35f7b989c139414
    a89d77b976c7f78496f7d536467c038a0af410df a8eedab83f6bc7d59d869a4c956ed94e0012153c
    a94c5336b3ec3100f5455342068c5f381cb0e6ba abe47373dea4ce7aa68f442ba29b8981d89950d6
    abf25e735237d960746971b4bb029e6f6a32be4b 76370ee446918f3c1aeeae6c347f0abb016375ef
    ac83cf8862b304a000fc7095b552bed637a72d8a aeb3a14dadb80850bfa9556ce39c66341f7326bd
    af58f9fb25d94d870bacaad845077faf743d2a80 b0ad22778f5d59d32ed2a91a7f85cbd53785621a
    f74b688c972ab176918cd5b07e30f5337d4d5656 b1fd3a7cadf3edac4bac2f7074edc92c56df2cbf
    b4d3acf07cbb51098d78fd161d98a6bb2b42b2af b5b0ccd8f042b5e219192e4c4abc5a695bef4ba5
    add1fdf25a7d7b0dedb01240c3c275c7ecc66c99 b883aa24b864fe3386e6a71594a0301954a534e6
    ba164535114b375dcb66b77832d33a77d60ef1d8 bb2b68d3fa91ec2f117573e2af0ca00e6c1f5a99
    bcff098c34f28a043b4ad36777808c90ea3eb436 bd132685b528e2ad31c025066906c8ef9577589f
    bdfa674e4d7b7194451fc03db61331e7fc70440c be8f98485373dd94212141ee3051866565e11008
    bf2f67f32a03cd1b7d82d082b8a1a8920f6c6099 c01895e635452e1e6468590fa2cf51f9c0074d63
    c1cceb393c023744932cf2c6750c79fb9ec279cb c2365ad61c54d67d4aa16e01897ea29e24c4eda6
    c247607ee0c61bb18c8ddb949471b46823bb0e85 c35614cd5f8c2cf303d34d6976d7bccd2b2d6f5f
    c573792330aafd1ca8bf957688e4ac13908d9511 c8b166d1348a4c0f591d3a88da21931343beff33
    cd49e69631781ca2f684942b1cd247beb2c3f07b cea1c55ac5898b7a0b27ae324c98fb190dfbcf9f
    cffb961def14b3f24ae14ac315480d39ba8c245f d0e3d7c0dc4aaad950e9d471a25f4df02333d49e
    d5dee5aff03233a2db8fca029e8a0042ad747972 d6aa06f3a3d5b0eb56a86b0cb2b907f7e12e1718
    d76e2705a7bf2b31bf65a70ed92b3c940dda8783 d7a31323ba294cb3e6090cb203917a399bea5071
    7b97a083f1d65756fe3e50326e40685feeb97065 d95f9972eb8a6eaada67c73692a35b0ccdc2602f
    d996f3977a150e5a5a9c22770a3ec811f45c0af9 df30d78f527b8e63f939afe8e96b1ad026fc2ecc
    e1436abc8a5fbc9b55b749d21a909b3a015fd59b e408c048227cf54473711bab23ebdac92917a949
    e6ba7bca77e93412f6c8fed51b9fd993fa03ceba e733cd9a789acd6e26bb1b9d6fbb07eeda31cba3
    e76fed7531052caf286e0bcfea4b977a23c59214 e77156eef90199f1d4c1615c37f6d4b903dbfa53
    e8c5c2f917ec72a57c254fe56d5c5ff7461991c3 e8e39a7666dd9ae0899c3d09613ae6a77700e3ed
    e91c077edfb4655a5be3723b390c4654c38a1d0e ea062b50c2bea01c1a4f1eefd6909a04c288f8f5
    ead10f399fe2501f90ffabcff968c4dca771dc44 ead6b6cfa6c2863be0a46de116e08dceb5c2530f
    ec2a0fef8bbd6c787cc0274765ff2b4d48b88351 ed4c6cca8e700fcd473258c3dee08cab821dc5ad
    ef30136fbd014e5ca07bb610fc2665af6bc6199c ef6b4ae37cdc48e95b839912eb83e48c840185da
    ef9f8da7bebc9c0a2463f4c3f956e43ac30c2c0d f0de8522fa8134ddcd6be3066a0dd05759fd170f
    f1bbca1251ae183fa89603b0ecd860cf3a2f144d f26930738a6635c20ac56baf6a033a0d4986459d
    f3036eb051d881ea7a49a2062704ac429c850ca2 f8134c6d0a01b6c819d46f2c3d28b689f5f169b3
    f99c924f26d76e21105fa2b506192dcddfe90d05 fde84a2622a710be11c35023fe0963641ff2ae41
    ffd6963dd00cabb53ce437e426cf18d4b205929a
""".split())
SYMBOLS = tuple("""
    CircularBufferManager CircularBufferSpec GatedResidual GroupedGemmaRMSNorm HyperConnectionBase
    HyperConnectionConfig PLEShardOverlap PLEVocabParallelEmbedding PleShortConvAttentionBackend
    PleShortConvAttentionMetadata PleShortConvAttentionMetadataBuilder QSACompressedKeyCache QSAForwardMetadata
    QSAIndexer QSAKeyStateCache QSAMetadataBuilder QSAStateBackend Qwen4ExpConfig Qwen4ExpDecoderLayer
    Qwen4ExpForCausalLMConfig Qwen4ExpForConditionalGenerationConfig Qwen4ExpMTPConfig
    Qwen4ExpMTPModelArchConfigConvertor Qwen4ExpMixtureOfExperts Qwen4ExpModel Qwen4ExpModelState
    Qwen4ExpMultiTokenPredictor Qwen4ExpNGramEmbedding Qwen4ExpPLEFp8EmbeddingMethod Qwen4ExpPLEGroupedNorm
    Qwen4ExpPLELayer Qwen4ExpProcessingInfo Qwen4ExpQSAAttention Qwen4ExpQSAFlashAttentionBackend
    Qwen4ExpQSAFlashAttentionImpl Qwen4ExpQSAMetadataBuilder Qwen4ExpSparseMoeBlock Qwen4ExpTextConfig
    Qwen4ExpVisionConfig _QSAStateCache _apply_norm _build_non_spec_metadata _build_qsa_metadata_kernel
    _build_qsa_metadata_torch _claim_ring_block _compress_qsa_groups_kernel _dequantize_embeddings
    _expand_qsa_indices_kernel _get_embedding_weight_scale _get_mamba_spec_for_layer _get_mamba_state_copy_funcs
    _get_packed_kv_cache_groups _get_ple_embedding_quant_method _is_deepseek_v4_eagle _is_prime_64
    _iter_qsa_attentions _logical_positions _logical_to_physical_qsa_slots _make_draft_vllm_config
    _make_layer_multipliers _make_vocab_layout _norm_rope _nth_prime_after _ple_conv_kernel
    _ple_conv_writeback_kernel _ple_gate _ple_gate_fake _ple_gate_kernel _ple_ngram_ids _ple_ngram_ids_kernel
    _prepare_ngram_context _qsa_merge_splitk_kernel _qsa_mqa_paged_kernel _qsa_pre_indexer_kernel
    _qsa_sparse_paged_gqa_splitk_kernel _remap_ignored_layers _remap_mtp_weight_name _remap_qsa_cache_scale_name
    _requires_qwen38_tep8_emulation _run_qsa _shift_apply _shift_precompute _short_conv
    _short_conv_dilated_decode_batched _short_conv_dilated_dispatch _short_conv_dilated_prefill_batched
    _short_conv_dilated_spec_batched _short_conv_fallback _should_replicate_misaligned_shared_expert _splitmix64
    _store_qsa_rows_kernel _strip_qwen4_exp_mrope _supports_fused_pre_indexer _update_and_compress
    _validate_csa_linear_tp_layout _validate_mqa _validate_ple_config _validate_ple_layer_ids
    _validate_qsa_config apply_qsa_rmsnorm apply_qsa_rope build_qsa_metadata_triton canonical_qsa_rope_positions
    circular_qsa_slot_mapping combine_and_mix compressed_qsa_slot_mapping compute_ngram_ids
    compute_ple_shard_overlap copy_ple_embedding_shard_ expand_qsa_block_indices_cuda forward_qsa
    get_gdn_mamba_state_dtype_from_config get_gdn_mamba_state_shape_from_config get_mamba_group_ids
    get_mamba_specs_from_config get_mamba_state_copy_funcs get_max_layers_per_page_size
    get_ple_mamba_state_dtype_from_config get_ple_mamba_state_shape_from_config get_quark_ocp_mx_group_size
    hyper_hidden_size is_kv_cache_tp_replicated ngram_context_len normalize_compressed_keys num_groups_for
    ple_conv ple_gate ple_ngram_ids prefix_cacheable project_qk qsa_compress_groups_with_ratio qsa_mqa_paged
    qsa_pre_indexer qsa_select_paged_tokens qsa_sparse_paged_attention qsa_store_cache_rows
    qwen4_exp_amd_ple_ngram_embedding qwen4_exp_amd_ple_ngram_embedding_fake qwen4_exp_compute_ple_ngram_ids
    qwen4_exp_compute_ple_ngram_ids_fake qwen4_exp_ple_short_conv qwen4_exp_ple_short_conv_fake
    qwen4_exp_qsa_with_output qwen4_exp_qsa_with_output_fake refine_fp8_moe_block_shape short_conv_layer_ids
    validate_mamba_state_copy_funcs widest_group_bytes without_modelopt_fp4
""".split())
SKIP_TOP = frozenset(('proc', 'sys', 'dev', 'run', 'infb-buildcheck'))
HEX2 = re.compile(r'^[0-9a-f]{2}$')


def run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), input=stdin, capture_output=True, text=True)


def object_dirs(root: Path):
    """Every `objects` directory holding pack/ or loose-object subdirectories, on the root's own filesystem."""
    device = os.stat(root).st_dev
    for dirpath, dirnames, _ in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOP]
        keep = []
        for name in dirnames:
            path = here / name
            try:
                if path.is_symlink() or os.stat(path).st_dev != device:
                    continue
            except OSError:
                continue
            if name == 'objects':
                try:
                    entries = os.listdir(path)
                except OSError:
                    entries = []
                if 'pack' in entries or any(HEX2.match(e) for e in entries):
                    yield path
                    continue
            keep.append(name)
        dirnames[:] = keep


def audit(objects: Path, app_store: Path) -> dict:
    gitdir = objects.parent
    owned = (gitdir / 'HEAD').is_file()
    g = ['git', f'--git-dir={gitdir}']
    scratch = None
    if not owned:  # a fragment: read it through a scratch store that names it as an alternate, then remove that
        scratch = Path(run('mktemp', '-d').stdout.strip())
        run('git', 'init', '-q', '--bare', str(scratch))
        (scratch / 'objects' / 'info' / 'alternates').write_text(f'{objects}\n')
        g = ['git', f'--git-dir={scratch}']
    rows = run(*g, 'cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype)').stdout.split('\n')
    rows = [row.split() for row in rows if row]
    report = {'store': str(objects), 'owned': owned, 'objects': len(rows),
               'withheld_blobs': sum(1 for oid, _ in rows if oid in WITHHELD_BLOBS)}
    reader = subprocess.Popen(g + ['cat-file', '--batch-all-objects', '--batch'], stdout=subprocess.PIPE)
    grep = subprocess.Popen(['grep', '-a', '-o', '-w', '-F', *[a for s in SYMBOLS for a in ('-e', s)]],
                            stdin=reader.stdout, stdout=subprocess.PIPE, text=True, errors='replace')
    reader.stdout.close()
    hits = Counter(line.strip() for line in grep.stdout)
    grep.wait(); reader.wait()
    report['symbol_occurrences'] = sum(hits.values())
    report['symbols_hit'] = len(hits)
    report['read_errors'] = int(reader.returncode != 0 or grep.returncode not in (0, 1))
    if scratch is not None:
        run('rm', '-rf', str(scratch))
        report.update(remote_refs=0, refs_off_tip=0, commits_off_tip=0, alternates=0,
                      unreachable=report['objects'])
        return report
    refs = [line.split(' ', 2) for line in run(*g, 'for-each-ref',
                                              '--format=%(refname) %(objectname) %(*objectname)').stdout.splitlines()]
    report['remote_refs'] = sum(1 for name, *_ in refs if name.startswith('refs/remotes/'))
    report['alternates'] = sum(1 for name in ('alternates', 'http-alternates')
                               if (objects / 'info' / name).is_file() and (objects / 'info' / name).read_text().strip())
    fsck = run(*g, 'fsck', '--unreachable', '--no-reflogs', '--no-progress', '--connectivity-only')
    report['unreachable'] = sum(1 for line in fsck.stdout.splitlines() if line.startswith('unreachable '))
    report['read_errors'] += int(fsck.returncode != 0)
    if gitdir == app_store:  # the agent's own repository: its history is the task's base (no_withheld_code.py)
        report.update(refs_off_tip=0, commits_off_tip=0)
        return report
    head = run(*g, 'rev-parse', '--verify', '-q', 'HEAD^{commit}').stdout.strip()
    tips = {head} if head else {oid for name, oid, _ in refs if name.startswith('refs/commit/')}
    report['refs_off_tip'] = sum(1 for name, oid, peeled in refs if (peeled or oid) not in tips)
    report['commits_off_tip'] = sum(1 for oid, kind in rows if kind == 'commit' and oid not in tips)
    report['read_errors'] += int(not tips)
    return report


FAILING = ('withheld_blobs', 'symbol_occurrences', 'remote_refs', 'alternates', 'unreachable', 'refs_off_tip',
           'commits_off_tip', 'read_errors')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path('/'))
    args = parser.parse_args()
    app_store = (args.root / 'app' / '.git').resolve()
    reports = [audit(objects, app_store) for objects in object_dirs(args.root)]
    for r in reports:
        print(' '.join([r['store'], f"objects={r['objects']}"] + [f'{k}={r[k]}' for k in FAILING]
                       + [f"symbols_hit={r['symbols_hit']}"]), flush=True)
    totals = {k: sum(r[k] for r in reports) for k in ('objects',) + FAILING}
    failed = [r['store'] for r in reports if any(r[k] for k in FAILING)]
    print(f'dependency store audit: {len(reports)} stores, {totals["objects"]} objects read; '
          f'{len(WITHHELD_BLOBS)} withheld blob ids, {len(SYMBOLS)} symbols; '
          + ', '.join(f'{k} {v}' for k, v in totals.items() if k != 'objects')
          + f'; {"FAIL: " + str(len(failed)) + " stores" if failed else "PASS"}')
    return 1 if failed or not reports else 0


if __name__ == '__main__':
    raise SystemExit(main())
