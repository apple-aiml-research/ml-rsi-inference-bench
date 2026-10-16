#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private build audit for the withheld relative-attention and model-kernel code; bind-mount it, never
add it to the agent image.

Inspects source, installed code, bytecode, archives and caches, and every Git object's content rather
than reachable history only: the working tree and the object store of /app, all site-packages, /opt,
/public, /root (caches included), /tmp, /usr/local/lib, /var/cache and the checkpoint's text files.
Checkpoint tensor payloads are excluded. --filesystem-only skips the Git audit (for scanning image
layers before their whiteouts apply). The withheld set: the landing's in-tree FlashAttention-4 fork
with its relative-bias paths, the model's CUDA kernels and wrappers, the landing's Python/Triton
kernel modules, tml-fa4 at any revision, and the repairs' content (FORBIDDEN_BLOBS: every blob the
reconstructed closure trees add or change relative to the base, minus four that the base's own history
already holds). An unexpected copy is a material-audit failure.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import json
import lzma
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile
import zlib

FORBIDDEN_BLOBS = frozenset((
    '00b8b1495b5690d6b0d786a26a09eb1879fc3f7b', '01ce30b9e7f73b62db8284ebb66e58646aafaa85',
    '020995054c2e5932b0048cbef56a4c876d74a1eb', '021d222834aa7cc388a8bf253d2d811917331086',
    '0263d6e8477dedc9905d17869e7aa82cfcda57a5', '03177bacb2f04ce8cd7470eb241a24ccaceaec0e',
    '03598802d7e692ba87d1ef593bcee2d40e89ea28', '03a9eead214d78764733ca9ec7a2cf546ad85607',
    '0509cb29b7985d57b4cd543eb7c0e825f4b1199b', '087b1293c492c6b6734d0bab947b372219e7f2a6',
    '09062c8969caff62b1da6b05632713ee836787ba', '0a04956065f524fd5e522521187ca630979af1e3',
    '0c1d582877dbf615faf96b4dda9ee268c105997c', '0c797b970d75dfb31f4fd33fe662fce8b66244dd',
    '0ca99255e4a4f468312901d52a8d3b107dac08d9', '0e6d0f214ee7e5a312473bbc459a815f9a2cc105',
    '0f424b38290bc6237fdacfd2c5f4b120f92d7fa0', '0fe41032ca2ad764a7a0b410b1af13f0f6babe93',
    '10d08d628e8a1c7faacc873af3927530937f72ef', '11a4e6531e6493fffc572cb386382c5339a47a95',
    '121811e82a1890487ff49e7b9b42aecfa2048c35', '126cc01eea1c246571992b359f9f7a3bc4abcd9a',
    '129224c7efb633ea42ee2c71e6ffeb1bf6714079', '13246f9c836f7a43fd4f20facce0fea57fe4bd5d',
    '13cc1cb7852ac5eae8e316e28a0f304f1e684fec', '1441af21fcae90608e4ad508546c4493e84f297d',
    '1502cecb74e3b91073fcb2eb2af74462ee8824c4', '150373a6bc3b6da10a68f1ac91f0d916b080ed0b',
    '157f0178f76cfef5527173824e5dfbee3db3e455', '16459ea7873931bdb52a85f102fa88ffa944213a',
    '18260ff3c16af73122505dcf9d1aa06d78b2fbf2', '1890dcc78f9afa226ebd26fe4806c535ca5528c3',
    '19119bcf331d79d38ef2025b434345ee840a5b75', '1a09e038597ac60ee28c44ca0edf4aa778efc319',
    '1a99d45e3ee26f58ae7f9ed11b7a395d338e482a', '1bee164d0f8b49411084f2a3e521b5851680f0a3',
    '1c4a4a8d6f9f6a2b8516ad31014c836dac544b59', '1d8bbea687e1ad6840845ee9fa95c94d129aec81',
    '1dd0fdb01500789853545efce84e183f799da078', '1e1a38997490ee0eb625a63d5d6e3337b323831d',
    '1e48004a58482070d150d5c2cceceb7a5c9baa9b', '20ef0183d3b688b1d49b52b0f9c44344765a0c59',
    '235952647cc30b622ff988f56b8f4a336f110f7a', '23fc6c0ef849681e47f646f1ef0dc77bb1d7cb1d',
    '24449924d918fafbb8346d46e8418cf648811614', '245e0eb1639eced9c82e4673bdf8d47f34f69cae',
    '2491395f3824ae3752e20e1fe8c1d3489089c968', '24baa18d57538da5b9e85025c8a631c7ea94235f',
    '25d5184a3a2bd547dbc74c2af934652f9a8a9a29', '262dd0508d6a0e52ba2a4fc5ea85dfde098e7792',
    '26469b07315b81c78f065fdc7a840dec3c33e2a2', '2667e03e8267eef3b7cb4667d43de31b8ec67cd2',
    '293d461a1469e94634932d0172c2d289a5a5e2a4', '2bcb12d3678c0eb29bf82f38fd79dfafdc904332',
    '2bce9dc31c16d4b47d4011d602a399563ee70fce', '2bff25e5ff6a1775f2c930d206aedf983bbe88f6',
    '2d5791ec1c1ce4e5991a880a5cb6d6bc83b1f28f', '2e88ff8ec0cde3d1c4d585938ee61459bddcf414',
    '2f69e6387ccb9938f8af0636784e9a8c8a1df1c7', '309efdc308f1096f5c75eaabe0ade643093c82cb',
    '329d71b317a6cf93ac7d0cc24320c8e7e8e2e8ec', '32b3fa8b5212f63dbe479fa5750dc00380f283eb',
    '33c685caa0b6fd63dc5e34234e0006b0a3bc8e7c', '3442ba544f7041875ade8d24839c6c6807a11da7',
    '37a9aa5b5094cada85de2196997d2a15fc0dc96f', '38df069fffc66ac1c4a74cd3130b0d152372be9a',
    '3aae353d2bef014c53aa2bf40e8e7a600153d78f', '3ac8bc241b909e9cca358195a83f67a55fff3d25',
    '3ba756c7da29c8c311c183d09b3ed37beadc1c15', '3cbc9516beeea00e059b3ec1041349b5ec581c12',
    '3d3ef92f383268bc24f428e36b14880259b65bdd', '3ec9c786100f23e09c9ff0b9b31915ef44f39a37',
    '3fae863398105bae395fe3b34aa41f5c2a26ad7e', '432eb9c43d12f4ed2f4422ecb4dd3df07b66f479',
    '43b196a7b9640cebb1a7c9f326533f02620ee260', '45119da5672cbcd907c0458f9a239c7e60373537',
    '452ee54410cc0ad2a23672f71c0900c372d7d174', '455bf5379b3b9e224f5d60ca36fea8d57a863d3f',
    '4583258b391aa70b1ecffbc4ff2d5d312b1a3ddc', '461767ccf598b4cad918ac61a22a35124258be39',
    '462c6f2154dffd7329291cea686bfa16bba7aaaf', '46ccf054c86bffddc1c89bbc973d88f3d39c29af',
    '475252077e3461d85cbcb94f6643cd5f45db4551', '47c7ee07cd4903a2d0a1e95e332b49cc111a4e0f',
    '47f6274b3e4949c4e240000eb68f5f00684975a9', '48b3e61488676d69d6ac84babfe44843cfc3ca0f',
    '4930ba563484cccc5cd134f97b6a10df21d17a5a', '498539cdad7793101faf05a3a988d0f1e7aee15d',
    '498cf4787aab2cddfab79bb073fb19a44606d5b4', '49d7c8c150238f777aeb6485266706d0c1eefe72',
    '4a19da4c9c7b319b47b11bb53ae8a459513e8f71', '4b831410ae63fc0089640d23d89c8836472a4b70',
    '4c44e06dd0378b612f62be9a3438966eadfbb8b7', '4c456b8c9c1ddf6b1b569dab297db653d49618c0',
    '4c6265ed6930233d7243f21edb381331496ea0b0', '4c67683dbee69417e34457370430e0fcc0252ed1',
    '4cf2e9f6a8c6e908eac326a6672118b3de64f856', '50a00b3e5fd4f7acb1deaee4c9a5cc0261914093',
    '534a430d765248b47c3c2c600532a0c09f9bf5bd', '537cc2e858bd41132b68a51acf283227d7c45fc2',
    '547a1a04cc9de9c88baab71e5b74c177dd8e25f2', '547a91290c7690ac4b544cf5a7c85e0712761b00',
    '582e3783db7307b735c043a0491686a55f168f84', '5b406127b78d806d3ae24072472fef031e8fa32f',
    '5bd4a56ac81fa1f5a927b4a7a2a8c1be31328127', '5c77f22a7b61bc1ab75aa0337e0cd29c5df60bc8',
    '5d08d11adbcdeb1db7aaa95d5c86b1f4d29328ab', '5da717c7d69c06383ac9298cbcdbb98b0a38a0f6',
    '5e5681d2593602881aac73d90232954aee63c77e', '5e9676993c15cd650e1c52c13de5a7357a72fc70',
    '5ea45ccd7bc6235fccebe548c86ba8d2183fe62a', '5fb2b0dcc689efc8f55d41de1c888a3e36b17d08',
    '600ee4cbb351f313cadb64df35c587c1b69c4ffe', '605cef4678a6208ed4b61254725e73eadb83cb61',
    '60c702d19b432c94797518fe27548dcee273da74', '6112e2c86a82b30f9772e855a3ccb73f3bf0582b',
    '61b20967f4625819b776621488f0214ed6518210', '63189cd5d65adba3eb303d8db8757243ced1d0a3',
    '65faa9dad01beb54f1088de94d62592676331f9d', '66758afaad0bb5998796c44326e9b27cc7374d21',
    '67e4d95e3701e51893a67e1c66e45abad2b36949', '68eccc0be97ccedc2e2acf8c404084fadd445032',
    '693cdd4b8ded13787b6de26520fc51197551fec1', '696f9528f8e10c736021d538c6099bca3e20a8f2',
    '6ab7cd2722557b1fb6e1d14c56140cdc29eafdfa', '6ac9f4c14cf25d9d74a85c541a79891c033ad22c',
    '6d09e81332f0c1541d2a2b66f9a18b6caa6a4ee8', '6d56085279c36264aa23b8882685b4513ef83790',
    '6e528bf8866f7b40b908d654e4ded70eef4f1499', '7467cc9712952841e61648cc90f9c671f4e1bc58',
    '74999b4ee35a65dbd3555a1c5c46802fa9eab0c7', '765264d750015a7006b75cbe215d2766fa7c461d',
    '7661ea6e823965ca0456bf359ad50cb1080260e1', '76e2a09216971b80e4ecdf0b0f3cdefb76227a50',
    '789dbe713bb12a67c2fae541457020ac75815945', '78dc4adfde8bb997320f8e1a0a7781ac168008c8',
    '797b12f42e1035e4b94bb516adc8e599321b8923', '79f3e612f3c44fb4f0a877b885af2a4f6083e0f2',
    '7b3ca52e5640afe62352d2e638e9acb109aad42c', '7d0f79dce35e0f3e2b908c5364fafca3c9662f0a',
    '7dfa25925727886bfcab63f6c80b3fb266055b29', '7ee8fbb2a7c7e66c7a78baf26b7b1189691640df',
    '7f5d0e0e21f6bba56632c88edb8673d1d7bd8853', '7f66873f58343e4fbb4ab7e798f376a1d4157b6a',
    '7f821f1a0350acb6dc4d9b441134866a5a4140b3', '7fc452ba6edb20de3e39735876487e05c0134621',
    '8061a092b3b65928b9ed73e33a2125c538e7cdf9', '814442e401e20e13e7564c7fbfa25ef89c620c32',
    '8154c14b216ad0187fb0f5147e76971f00ebe833', '824698b3b3abefd8e07cd851b49b6b9f9e606027',
    '82f454c31284b65154ff2d946c01dddd0146a1cb', '82f5ffb6ae1b662e210522338bae701d353636eb',
    '82f74c02d16b88014529bd9d3be52a30be0909fd', '8405b0e69bdbf411771a3396e7ec959a285d4d0b',
    '840654ae7c505dd02b92670042f7e5807ac5f996', '841000079e49edf99e32fa9ad6f528ecc4400282',
    '845aa47dc667ce093da81299f26d080b081c817b', '846d06fbd2caf7149eba467c4e4dbdd82d6d1141',
    '866c1ad24af69de78db8c087b8f687bc5f9ffbed', '874e4a49390b4822bdf23d2ac7aef8642fbc7cbc',
    '87cd13ad55f8f6d04880fc5047e7d88e5507f1b5', '88df4efe04a04c41222b9999d23013a265c6fc95',
    '8a84701a6e5d5574a74757c0266acc28ecd7d67b', '8ad576f622a0e94c33795def0fdf6cee493e2b10',
    '8bb5019340c915f27ffb02dc1ef627f2f4485649', '8ca79335791d22f73591ce854e0e8b54cf9f7ba2',
    '8e5cac41c13edf7e4409812489404fe607672cad', '8ed7c7ac9af742d660bd5cdc68bfafc47222fea6',
    '8ef2c7ea6a6745d0b512517cdba55ed387cca8bd', '8f5884777207ea05b408c3e87f7b3a62afcce83f',
    '8fb06959fc4be3f3e8ea3e7d6aefc97e1d46f48f', '926d47e74b828f85ca8e1721467d13a3dbaa00f6',
    '92d9cea5810e055d6947b9b32605378bd5b09a21', '92d9dd04a1a82ac1a7cba435fb153151f2af307b',
    '93f4dd044de263da738948343342394b7fd5d94d', '95217d6b399e0dd1b215760d97235085b8818288',
    '955339cacda30bed6f4bef84be6812ef5ec5fa1f', '9675c604e73e4af50fcfded8a8eb0b65c637aa93',
    '9746d9a5c49ef90d921beef49a28f0bfb80be458', '9a3ac354075f6bc14e2950cde0d7785018cbeffe',
    '9ba26b4c326d6519749404f5aec1ecd8b65edc13', '9c47af75de79504b95042bc6cf39498b6eba0eb7',
    '9d7f09f08ffe98900f5d13defb1f461444c194aa', '9e9d725e1c4462f2c7e9a8875b32fba3112df4ff',
    '9edf4be19177af049eae28a95e583fdd8ab4acbb', '9f1303464454a31845e2883f0904399380cd226b',
    'a1375a814e40cbe056c99a323c9797f2d47ca244', 'a267dcf7b4a10ba9eedff38a3f53d63f1f556952',
    'a3121dceae0511718952e65cc9ecaeafd04c19a9', 'a65a3401806548a818846e4055b3a481aec04199',
    'a6b5495e796bf5f3a13042f8aa4ec83e9435b0de', 'a6ce3ab3c577e1be59bfef6ceaa35c60df4ed87f',
    'a77efdc78581cc544f402cee14f9a7f297e68748', 'a7ba69ecade739b3aff3105a38f25a1ef7bf5402',
    'a9474a8094c1613a0c6f3dfc7db5253f1d35af4a', 'a9a790c5839edc8b2e0e2ab55b5b4b3a2a36effb',
    'abe29dabd8e9920767ebf287efbed8067cd501ff', 'ac3855a3181428b6ef89b2722421563ff0d8a028',
    'b326e93abfd89e3eeb25ea01a119e46f9349d921', 'b472cb2263958f9bd19049c7e43b689a97148735',
    'b483d13b0439524cde5c30a26a915e5f4f996586', 'b4f24592e40abda3872f14591b1fff10e80e580b',
    'b53a48ee0110bc17ddd59139cf0d300237d3efcd', 'b74563752d80df87261eb555d2225f95bcddfc47',
    'b8dcbca3ed68ac4da84ebc3ccd9808fec3ce2be4', 'b9c3015cf3ba3b1d2a228a6ed275aae67e7860ac',
    'badc7a9ee391043357d1dae7bfb36ab64e322dd1', 'bd3c3c7afab88dd1acb18e403620c3e6b828525f',
    'bdd9542d7cadee0b701293081670eafd9045328b', 'be32e149b85557dc7dd25feb0f90f000a6f167ad',
    'bf39911577ffbd2901ac1a9fb2b0e9a69ac0efb6', 'c0366048b35d9672a33a0ce307638451b0543401',
    'c154e1c5581b28a15a7dcaa187351e0f69dfbe7e', 'c1cf5ffdef990019278178acc42c430cb1c98d04',
    'c427ce41e36a7fe009932e1521c05869af145560', 'c46a8c59681222ede0e8314a3589276ab76d2428',
    'c595d6757b7b0c2caa77fdab82b260f4906f3e61', 'c5ad825f37b5efa944c36af86669dc34b46bc327',
    'c5e77b6f824d97db23817d66405d5eba8838e3c7', 'c621e8e69b1f533c75584eba01c0cad0f1909e3d',
    'c6485abca01bbdb23a4acfa96baa5af3be9d46d8', 'c719db663b1ce49f69a1146166f05b66d7acb828',
    'c7da9e70d44a23b3709a2c786bd3906bf1f3a607', 'c7f1b32ebd09cd70c92022a464b761693a55c466',
    'c99f559f44794e8c3a41c302dfac9bc8e2567c9f', 'ca1b753141c6cc1210c444ea0b179498f45858ff',
    'cbbd393172af450c7f53933a4109d31da586070b', 'cc8ae4e21badf2b22228bfd1b11ddbb13e8541dd',
    'ccafe35fcdfd6ec15a32ab02c38a6044ed1390ad', 'ccce228ef4d54380997569479d0cb5a956569e64',
    'cecb81ff510466907851b4327ba2e52d0144b556', 'cf68977c388e194afedc58a0302b80e5fb02c803',
    'cfb620c09fec1ea1edf5b70b9295d914eeed01bf', 'cfd9cde09ab93201281b3517a8fc72282d35fe54',
    'd24c95be777dba0207c77f4a9a00d606b403373f', 'd2fdab6ca24fcfd49c9c12d46fb9ea3b710dda72',
    'd39c20fbb1347a51f393c108447fd7e8b25706ae', 'd3cc5937d64c0821bfb007eaea16c7ac5a293a31',
    'd5ca7bbb104d5a58832b5af5ac94f9ce4d8e9245', 'd68e52457f1e960150884a27a3f41a1dbd2a748b',
    'd7690c00d57dac8503bcc6abd889fdc3c43d4dcf', 'd7ae9973eff610b8d8f49da6931a03fc3f784c2c',
    'd83eff63188215472038522cafea01dbea2fe7d4', 'd84c37929e1ff5e56078f47383d200afc8bc4559',
    'd8e1a681f6183bed60cf5414bb3476acfd52a0ef', 'd955ceedc9daeaae25aeef21c1dd6e4240579945',
    'da0a16b6c009996745cee860bcd1e7aadbace9cd', 'dba1feade6aeb6f903a349674c475afc0a946502',
    'dbb1d187d555443969a338604abf9ae9dada5773', 'dd4bf04b532a787752027dc350b2fec490fed3e3',
    'deec79c8a041b6fa59462f290608e2171967e25d', 'df296379543d8895b9aecff5c80801757a627167',
    'e09a54b0b85765d6404240412199109c3855f067', 'e3e05b28ae0d4bbbe0a60ac994edf4f62fcbf532',
    'e44735d1a22125ff10a8fd97f5a19933e6bcadfd', 'e697342f4eb1dcf8db6d3a14d088b3a87fcaefac',
    'e6d52acd3b6ba57634b070659b125a7e87f1604a', 'e7e3c2f2813902e2fdefc9a6df1486793765e26c',
    'e8f19a289a0aad63067bc2e237d2a1db1d614ae7', 'ea8c4bf7f35f6f77f75d92ad8ce8349f6e81ddba',
    'eae52665b6f5172b8997e0360cc613273422a3dc', 'eb2d1741abe9970f68ced7fade8d2b3b845d5326',
    'ebefd9cfd9808b3bb5d3909a63926a973808850e', 'ec573ee76b123b7acb6ef992e5fd3acadd24c970',
    'ecc132ef1fb177203c04732539fc57e2c9a1b085', 'eceeea83fd26e67fad3c90d6b07db16178af5611',
    'ed742e139a7aa3f5ebdb7042e39a266adebf0039', 'ede6ce7a27fecedd0370c347a12b7226b73aaee4',
    'f318a292c825c2a8b21783fbe93ac4346a4cec7c', 'f53b66e944cf4c882ce84beae5dff0ef4358a1ad',
    'f6b75710ee324e7a917312c74f91323dae4bba57', 'f6ba6bff126044d693b701ce759ce612a65b5d40',
    'f8f6d81fa1ac4e20090c5d6d54edaf7bfd153802', 'f9979f3e8104eb5bd0ee185fc84944fc6abc4168',
    'f9be93152968cd6ba0ac553177cba1ba89c1bafe', 'f9da4af0aaf98672341dd83206e566793fe052c1',
    'fa359c548306da6a29337fceea03a5e9eb926c1d', 'faaa1ff1e3a3b99351cb9a1137ba36ed195e8ee8',
    'fc7d3507089a1fba5fe79ff1da8f49ef471022bd', 'fcb344fc038310ae8e9444a0c30b565b66abee52',
    'fd535dc7b313bcc2ef24b4ad2159c65aaa4c28ae',
))

MARKERS = tuple(name.encode() for name in (
    'ShearingBias', 'CuSeqlensToBlocksKernel', 'rel_bias_prep_cache', 'score_mod_rel_bias',
    'get_inkling_relative_attention_score_mod', 'relative_bias_score_mod', 'apply_log_scaling_tau',
    'inkling_attn_prologue', 'inkling_ar_', 'inkling_two_shot_all_reduce', 'inkling_multimem_',
    'inkling_rel_proj', 'inkling_row_scale', 'inkling_gate_topk_renorm',
    'fused_causal_conv1d_update_decode', 'fused_gather_scatter_to_sconv_cache',
    'fused_draft_extend_sconv_cache', 'SGLANG_OPT_USE_INKLING', 'SGLANG_INKLING_FA4_USE_PIP',
    'flash-attn-4-forward', 'colfax_collab', 'tml_fa4', 'RelLogitsProj', 'InklingMMConfig',
    'InklingForConditionalGeneration',
))
#: Where a marker is legitimate: the checkpoint names its own architecture and configuration classes,
#: and the model-author implementation supplied for reading names its model class.
EXEMPT = {
    b'InklingForConditionalGeneration': ('/model/', '/public/reference-impl/'),
    b'RelLogitsProj': ('/model/',),
    b'InklingMMConfig': ('/model/',),
}
PATTERN = re.compile(b'|'.join(re.escape(m) for m in sorted(MARKERS, key=len, reverse=True)))
OVERLAP = max(len(m) for m in MARKERS)
ROOTS = ('app', 'opt', 'public', 'root', 'tmp', 'usr/local/lib', 'var/cache', 'model')
#: The hub cache may hold exactly the base's declared hub kernel (python/pyproject.toml,
#: [tool.kernels.dependencies] kernels-community/sgl-flash-attn3).
HUB_ALLOWED = {'models--kernels-community--sgl-flash-attn3'}


def exempt(label: str, marker: bytes) -> bool:
    return any(label.startswith(prefix) for prefix in EXEMPT.get(marker, ()))


def inspect_blocks(blocks, label, hits):
    tail = b''
    found = set()
    for block in blocks:
        value = tail + block
        for m in PATTERN.findall(value):
            found.add(m)
        tail = value[-OVERLAP:]
    for m in sorted(found):
        if not exempt(label, m):
            hits.append({'path': label, 'marker': m.decode()})


def inspect_stream(stream, label, hits):
    inspect_blocks(iter(lambda: stream.read(2 ** 20), b''), label, hits)


def inspect_gz(path, hits):
    """Decode bytes, never deserialize joblib-style .gz fixtures (zlib, optionally with a ZF header)."""
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
            if source.read(1) != b' ':
                source.seek(-1, 1)
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
    for name in ROOTS:
        base = root / name
        if not base.exists():
            continue
        for path in base.rglob('*'):
            if path.is_symlink() or not path.is_file() or '.git' in path.parts:
                continue
            if path.suffix == '.safetensors':
                continue
            scanned += 1
            label = '/' + str(path.relative_to(root))
            try:
                if path.suffix in ('.whl', '.zip'):
                    with zipfile.ZipFile(path) as archive:
                        for member in archive.infolist():
                            if not member.is_dir():
                                with archive.open(member) as file:
                                    inspect_stream(file, label + '!' + member.filename, hits)
                elif any(path.name.endswith(s) for s in ('.tar', '.tar.gz', '.tgz', '.tar.xz', '.tar.bz2')):
                    with tarfile.open(path, mode='r|*') as archive:
                        for member in archive:
                            if member.isfile():
                                with archive.extractfile(member) as file:
                                    inspect_stream(file, label + '!' + member.name, hits)
                elif path.suffix == '.gz':
                    inspect_gz(path, hits)
                else:
                    opener = {'.xz': lzma.open, '.bz2': bz2.open}.get(path.suffix, open)
                    with opener(path, 'rb') as file:
                        inspect_stream(file, label, hits)
            except (OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile, zlib.error, lzma.LZMAError) as exc:
                errors.append({'path': label, 'error': repr(exc)})
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
    # Every object's content, not only reachable history's names: a marker in any blob, tree or commit.
    content_hits = []
    proc = subprocess.Popen(['git', '-C', str(app), 'cat-file', '--batch-all-objects', '--batch'],
                            stdout=subprocess.PIPE)
    out = proc.stdout
    scanned = 0
    while header := out.readline():
        sha, kind, size = header.split()
        body = out.read(int(size))
        out.read(1)
        scanned += 1
        found = {m for m in PATTERN.findall(body)}
        for m in sorted(found):
            content_hits.append({'object': sha.decode(), 'type': kind.decode(), 'marker': m.decode()})
    proc.wait()
    return {'objects': len(objects), 'objects_content_scanned': scanned, 'forbidden': forbidden,
            'refs': refs, 'non_ancestral': non_ancestral, 'unreachable': unreachable,
            'alternates': alternates, 'content_hits': content_hits}


def caches(root):
    hub = root / 'root/.cache/huggingface/hub'
    entries = sorted(p.name for p in hub.iterdir()) if hub.is_dir() else []
    repos = [e for e in entries if e.startswith(('models--', 'datasets--', 'spaces--'))]
    cache_root = root / 'root/.cache'
    listing = {}
    if cache_root.is_dir():
        for p in sorted(cache_root.iterdir()):
            listing[p.name] = sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink()) if p.is_dir() else p.stat().st_size
    return {'hub_repos': repos, 'unexpected_hub_repos': [r for r in repos if r not in HUB_ALLOWED],
            'root_cache_bytes': listing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'))
    parser.add_argument('--filesystem-only', action='store_true')
    parser.add_argument('--report', type=Path, help='also write the JSON report here')
    args = parser.parse_args()
    report = filesystem(args.root)
    failed = bool(report['hits'] or report['unreadable'])
    report['caches'] = caches(args.root)
    failed |= bool(report['caches']['unexpected_hub_repos'])
    if not args.filesystem_only:
        report['git'] = git_audit(args.root)
        failed |= any(report['git'][key] for key in ('forbidden', 'non_ancestral', 'unreachable', 'alternates', 'content_hits'))
    report['forbidden_blobs'] = len(FORBIDDEN_BLOBS)
    report['markers'] = [m.decode() for m in MARKERS]
    report['passed'] = not failed
    text = json.dumps(report, indent=1)
    if args.report:
        args.report.write_text(text)
    print(text if failed else json.dumps({k: v for k, v in report.items() if k not in ('markers',)}, indent=1)[:4000])
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
