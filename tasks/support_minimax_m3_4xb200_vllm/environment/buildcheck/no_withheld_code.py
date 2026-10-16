#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Private build audit of the withheld material; bind-mount it, never add it to the agent image.

Withheld from every carrier: the MSA package (fmha_sm100, any revision), the landing's Triton index/top-k and
sparse-attention ops, its fused QK-norm/RoPE/KV-insert kernel and bindings, and every other file of the landing
and of the repair candidates that the prepared base does not supply. This audit inspects:

  * the filesystem: /app's working tree, /opt, /public, /root (its caches included), /tmp, /usr/local/lib,
    /var/cache and the checkpoint's text files, reading inside wheels, zips, tars and compressed files, for the
    withheld implementations' identifiers (MARKERS);
  * /app's Git store: EVERY object (not reachable history only) against the withheld files' blob ids, every blob's
    content against the MARKERS, refs that are not ancestors of HEAD, unreachable objects and alternates;
  * every other Git object store in the image: the dependency clones under /app/.deps and their submodule
    stores, cargo's git database and checkouts, any directory holding HEAD and objects/: every object against the
    withheld blob ids and the withheld commits, every blob's content against the MARKERS (the EXEMPT content
    excepted), and every ref pointing at the store's checked-out commit (one commit for a store with no HEAD);
  * the Python environments: no installed fmha_sm100 / msa / kernels-hub package, no TensorRT-LLM, and the pinned
    Transformers (5.5.3) and FlashInfer (0.6.12);
  * build and JIT caches: no FetchContent fmha_sm100 source under /app/.deps, no Triton, TVM-FFI, CuTe,
    torch-extension or pip cache carrying a MARKER (covered by the filesystem scan of /root and /tmp).

--filesystem-only scans a root without the Git and environment checks (individual image layers). Checkpoint
tensor payloads are excluded; its text and configuration files are inspected.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import zipfile
import zlib

from no_answer_address import FORBIDDEN_IDENTITIES  # the withheld commits (bind-mounted beside this file)

#: Blob ids of the withheld files of the landing and of both post-merge repairs (every changed path at each
#: of the three revisions, minus the supplied files and minus any blob reachable from the upstream base).
WITHHELD_LANDING_BLOBS = {
    '012307758ca6fc6bd85147f300247d1d4108f60c',
    '021eb2ea419e9021cd24f482cc2f6ac7d8a32fd0',
    '031f8d1a0a283deef74dbba3bbee0958e5fab1ee',
    '0340ca9a477c1005818487c08f6b8b5c1ed76123',
    '04d6aa4778c3d5fb18ea51add5bd87c0929eb0e8',
    '06c8048cd900b944d19a9cee43d1c9f4fe88e758',
    '0755699d1a4545649e8f5af5de77bbf2c6b24fab',
    '0df2ed85bbd173288513103256b19a2f5dd683fe',
    '109189a033a215563665458ec28becf121052eef',
    '1246f1721c2084e5415fc7dcba855dfe4dbfc41e',
    '1259ec0c1bf7e83b9c947cadb09791c547b4ad0c',
    '13dbce5368f8c701a99ca8e5870a1e550547ff87',
    '15610552f23611d0c1a592075f2235ecda2b2e5a',
    '160cff1eeb54205052f9d5f39152b5d491452a41',
    '16e0df0df648c2380c274039497ea1a084125052',
    '18bf14249111714bc879c605c85b4bf9df48b8f1',
    '1b5030b190960dd3758a25d156389be749f31530',
    '1cc8e8167a622b0e9a6fdd93e409345415402494',
    '1f71e14cef7b689b41b575eb5380eb686d367cb9',
    '208adfffea51cc441b0abb513144c170af5bc177',
    '208c2d690067ab2fbdaffd65bdff363dea91666f',
    '21b5e7494d7b48604e79f99f3c7a18b37892730b',
    '21bda8e173fd9cd6570694ed6d4e824af51365d4',
    '2254843858650bea85c7acf1273d4af0171b7d73',
    '22c026d358180916e983764ddaad94faa7925eb5',
    '22d96f9de9796beab2930f849a03a8e9a600d3e7',
    '23b8b3ed319779dae240c66c1497eb232ac423b4',
    '24ec55e400634e254a3136e4ed1d5f93f0a85ec2',
    '26cd21a015d0a976f8c5a5fa2a2ea5a3ab327233',
    '28becf7bfec495f88f012a55eb563f6c0083bf5b',
    '29d46860d9d141b2c0fa822669c0965bec88e78e',
    '2aaa7dfc49c887060efc467c336acc90908d68a5',
    '2bdb26e1a8dd6e5d4da5ba038be80b232fc1d4e0',
    '2d8a32ef3d576820a043f87fcfb6f1df1b86a74d',
    '2d8d46cacb7f8857968ab0bc721873a74969ef5c',
    '31ef144e2374393985afcac6f078f49553680983',
    '3268d125bb2c9616b43b5b2c8bb821eb8684fb9e',
    '32a2d86899ccf066ec1d37473bb7af3df12a8714',
    '32b4bc97ede7e756a2b2e2a116a31d16f1d0cd4f',
    '33851fdc8622c177acc3d7d69a38ea446a401508',
    '364608a806a03e31f3dae952885b4d2258f4fdb1',
    '3878f3038bd2ecd25f383450f149199d416190a4',
    '3897f8e1b3f68d24d5e3df2c8e694714698c1877',
    '398ae1016f3a121cdc14a7c20249177fcdba7d01',
    '3eec58aafabf26c993365adba0b56dbe88558f39',
    '3f79d4f5db0ca6832bcb6686fb806d9d35e4e29f',
    '40287e166b2afec1b5b82ef1329c2fc1ed846116',
    '407e57ca2f940fa3e5a5330be4d373329088f4ad',
    '40f23b95d1076e45dde5ef1af4c2cdcbb4999d90',
    '4261b702cd74cb7c36fcc7d2b693ba25c9161d35',
    '486aa7e4054b765b3d03f66181e16a3ad4070a8f',
    '48c1902e29af04ca5d3e18f206e8bdf27bb18056',
    '48db183d5a3a43057bc72ac0084163c98212c881',
    '49ce174400611414909bbd4a9fe403d4c9284985',
    '4a2414f5b83c89bea1e9843abe15f1a752297079',
    '4a72b6bc2c91771221d4a129fd109865d368f1c1',
    '4bd9244472871db52cd9e4b8d7d29a174b73519b',
    '4da528056042b764a45003bae56cb03ec1320f17',
    '4fee50068e47897983537b4d1e962ef70420cd30',
    '52d2851e2a4c667dbd29b61ba92300fceeb10f34',
    '55542230885b4539eb8f21f95f0f986b5df57cc5',
    '561e4020cb1f401d41691b9c3b69e526fd40f4b5',
    '5681d12554f3bb4b758bf87b6ac29f26ef9eccca',
    '58ba7c8cb4030dc1d30cecf615bebcecb2e27bad',
    '58d987770c65b0b79fe0e2777a03c9f6783ccb43',
    '5974e09624de8c95bdda163e3f81e49f5d1cd472',
    '5bbabc13dd685dec03255f39307d02690e90c89f',
    '5dd610f28788547dca0ae882d2a12f1b96655ec8',
    '5f52bc50eb21aa37cf8bab1ee53772cb84c5bb19',
    '5fb28b1c765a692829da37687560ab006a675c9f',
    '61d2376abb8c8a795058e77a857fea931b0ad47a',
    '669d1d37690220ae390119f2c947554006c36119',
    '66a9aa86bde3a970a9d57ae5fb249727003cc205',
    '69d4e416dfa82031bc1094236db59b88e04ce7bd',
    '6a70510e6ff75937852c8ee31f14a0c284919dc3',
    '6ab59f8c4b57ba17db6d53c56e79fffc45f28630',
    '6d35b978c0d6fe9cdec258e8109f7fc25019c281',
    '71dd7634a69721f86eb6753920d21c892fc18bce',
    '754270e6525bab8e4b80914a1fd0805086ad8145',
    '7561aa071ac8faeed9ba4a353d1f94ea7cbb7923',
    '76c343b91b1690cfa475369d367d02f9a704a61b',
    '79b2fba3889ba2b8b6c7d2a1ae2cc066695f6172',
    '7b6fb73cba9ac2cd4e2c9ef90288db04b64fb1db',
    '7bc7f1de4b34a70ea7d0e9a2aa13619f9bb496dc',
    '7d46faa6de8517762a44d17a7ce079a945eb615f',
    '7d9a39a7a4b7fd08d0c9fcb4c15d1d9654420bb8',
    '7de8a9d5fa128dcc1a826e074e1ff50d958df355',
    '80319003da599df3d7b132a44f1c1508a8f8b4a7',
    '8188fd59609681461eb9b670fec95fa7aff0e982',
    '820ce27bc3dd6bcce8f6fbf6525ac3c557cd6f42',
    '836649b725b41140c5036b0dda5e4cb627337dc9',
    '842d2ff31882330a303bd33ba9d10c5425d30f3d',
    '866bc30a151a139a2c3a79868410502971cb0c37',
    '867f71b9bf647ec39a008da62f089f7ca9ee584b',
    '879a328644426e51e945c631591a889f3dc70b7c',
    '8a261c502edeba0dcb76c4c01f48d06b02c4a45b',
    '8c756e25702f12fb40a857a1cd0b1521316e20cc',
    '8cca0e8e2998b080e4d174c6f9b402c1949ac077',
    '8f7ea822642d973404219aab537561ad8ba09485',
    '905a9bea3c59ee3ef14a5acede345ffc2fd4a36d',
    '91720911a63f8821958b054d040885da43cbc8bf',
    '919d71fb8e8b16d33f1a5654305bc326e5c9f1d1',
    '931dbfb61eb9528a0b93e1e926804c2d48b8d209',
    '96729614f822b2758587f7a564a76e57f3c948ef',
    '99a481cf67bb5c75f6af7145d3d6e5aa1d24f366',
    '99bf8d91b50f3d1660316626091af26d78f67b22',
    '9a14edc42713ca0ff96e4c750948bf317584d1ff',
    '9ee3a231b91fd81e0d3ac13bc14e9b4f69334542',
    '9f3ac1fd79deb2a8b23cdc9038bc22e49ec08cb4',
    '9f46cbd24239e033319d68dc72d4c6fcb1bc71c4',
    'a3bfa56f579ece3915a67e6e94e7a37a34c90c7e',
    'a44ca3487462923b6bcc8f57f31585ba57950777',
    'a5d74e14e64d96fc12fe8840b5a1089823b184a1',
    'a7f31afc5ef8664c112aa274ae497d4121992393',
    'a8628448c44844577f4c96277eb7ec7d7f2ba535',
    'a94e084ab628026d63c94831d4497c3b27198c46',
    'aaced78ed7cd0d7ec8222e84f12ba93cc7a54f40',
    'abc98df80abd220fc8743494a8ae3842bca52420',
    'ac79ec1a577c8c5fecb14ffe9ab15273f715b965',
    'acbf2cb46ad42927fa344363059fe37a970d132b',
    'ad40a7f18b7fe1fa7a8996159152138b31030c6f',
    'ad9aed07a31c888440548701ded4f7974b059587',
    'b3a7c2d9f6ec7377882d02cc6d5448e01a480e7d',
    'b5f0b80d0452dd5c00b73f0058d0d16fc1b612b8',
    'b80d3b8b3b8c732d56d1efb1250352c31df70e34',
    'b8c84ad2af2556286bda64e882d304b4ead33f6a',
    'b8d60e09e4b8a3c62abe6accd26b09f7e0c9c6db',
    'b98c74d459aadb056a5d41b9ee65f2c7a5fea2aa',
    'ba7d26c93b235c5cc083cf926be9fbf271ca0256',
    'bb3b6752472281e6ea0e66770c9426ec6ebeae54',
    'bd4393be5e7ce93fba4bfd5f9eecdc96cedb5ba4',
    'bdaa752a6032be8b6d1ea6f6963554ce223f8434',
    'c171b4bfe8c59caa81f377023eb2a56cc0ca5878',
    'c27ded3b83ef0302b4a95f9f094cf017bd37a63c',
    'c32ff38d998f81c12a05bbe4d0f7120638a81cd0',
    'c340dda85a6c523e9a63c582e67461533cfd4399',
    'c40500adc74aa2999a3522e517b462572a5454c4',
    'c6d9ed24adbf42062982a7ef7faaa24d3da8cf69',
    'c70200ed091486619615925c48198eb29e90a81b',
    'c74cb2d9a7b63e90b2262bf97987702c6b8f6944',
    'cb74877f6824d23efb66c410befa0795d2c9599e',
    'cbb1fa350f557774572a251653ba8cbe7bde7739',
    'd0d7c76481b0a315e9c57810d40394822f62594c',
    'd38014e468bf013257a1366f028018d585287e1f',
    'd4f2c1007b09d1856e71c313d3e36a38bf38e7d5',
    'd5144d76818a489c74ae24004815ddcf9f977a85',
    'd51a2dd312af9c68756003aab665a9c747875e2d',
    'd55c12d382a9ef2ae7abc15dfb539c1024a3874b',
    'd81458b3751423eec3a6feccd0c0cf77e73bce4c',
    'd9c041ba0b89b96ef2b3743351e0da9fdac459fa',
    'de505e122cf6310ccbd33952ea7372381e35b7d1',
    'e2c7f8821d9641eec2b116f5e802c03d8e613ea4',
    'e2cd14562c042c755f8b1b2e134da5d0000c146e',
    'e2cd62704fda07b33d194e4590af429bab817c51',
    'e43ad60914f305133aef574820427a4cf474eb05',
    'e4ece0a41972726f302c6ee946a3bc47135ace2d',
    'e5ae0fa7b69a43b5f4821575d769ecc41ffda1e9',
    'e6063b463284f912f6cd923d9cf65ff9515d56f2',
    'e66db04c22ef648752007030d257e70ef4d050b2',
    'e80224be70fa0d0115206c141b979d0419689b8b',
    'e95fe38d329917826ca7f1c070f91cb4bf01c42c',
    'ec75ce78bfbb0b040921568efd161d80ab178d6c',
    'ecd31f2dc01ab8dc608b49a4adf4cd38446caead',
    'f230b4d57909ea0cafababeb8e59e9bfef14597e',
    'f356ce6f4ff5bd2af32da923314cd48259c32ef3',
    'f360c83b8af204217de698d922467673edaee9a8',
    'f4319b99c81686df624430c9182f0c7b1880c6e4',
    'f47dcae310a47d1ab594ecc0fd8a8351c83c2b7e',
    'f62face1d2e11c61ffc5e4b3fe7f679dea2500f1',
    'f8b87fa860813155c6a13ae8444fd38385cced2e',
    'f9ddb2a9d2189ef2d4a97d5dca3d8b5b5f75c3bb',
    'fa6e902396f07fa3afbcf953c38316cf3b6b764f',
    'fcf05cf685942f51d140692856759accd0e1ec81',
    'fd1acabde2e53a367fa3c0b6590572a18a7dfb56',
    'fde1ba4f0c9eb520dcfa935ae97d5cfc30477ac2',
}
#: Blob ids of every file of the MSA repository at the revision the landing fetched (none reachable from the base).
WITHHELD_MSA_BLOBS = {
    '0201bf8a0ec2a07d6b4848bfe08299833726bc88',
    '026a6b6cd7835433152513eb6a281af93fe16c5d',
    '08eed56d6eb8418fe94296ea31f32acbfcd7fc18',
    '0a2a85ac222dc7003ab23dcc7da233658971831f',
    '0e41ce01b2dcb770231e2877bd1a9e18aa2c7e06',
    '0e5f1ae0da206970bc7cbc6020fb0cf89b025033',
    '0e8fd468fe9780bc0bd8f66b197a01bfb7000124',
    '0fdcbfa097c4f4c1dbc53862b78b8afec3a6da82',
    '105a433e4122553a5620a03686bafe44f879ed66',
    '10b2b1f667b25ae9a4beba003f7c6fc7f7421c12',
    '1250e28f978c5d5e501e5e7b2d2db54862699713',
    '132dd20ee3294d09a1c613a6bd4c8094c496fbf9',
    '152cf8a8faa48bf5589dfeea0b9e2ba18d0db58c',
    '15750948456ec3642104d52884f868e60ef04457',
    '15dfab1c4fc45dcec26ba7489b35624f5c46a698',
    '16d6092c9aed1aa32c08b7d9850c019dd535af95',
    '17e4475f7b46b09d9eb285ad7b699d6ba3c5ec4d',
    '18b99aea3f8b4915c03fe8147127374d920970f3',
    '19f6fc49c1f17a590aab319247ee7e8eb56cdfb0',
    '1d5753a8a175b529567e0be238f47fd4cc8401bf',
    '21c777e1e7f9285fdbd1f9e23a53c90a0085e4f9',
    '24d27298acfc8fdbab6cd10ef33f2afe1a8eb318',
    '27f711772f5c6fa16a86f4aa305f42a0ca9322eb',
    '28dc87e14b7e73cb199a96a90c07b9a2d2369f40',
    '2c5f6923c42a826d4f3dd1f192ce2fdb38eefbf5',
    '306b416b39e269cbcb86256958b53253534d1750',
    '361dab45ac35fd2da2dc83301a0de9b57a76ba1c',
    '3e2b84662377d5315717dcc8bff8d695bd192efa',
    '44507608aafc5c506826d10b97ee830784c4864f',
    '45542869f5b76448fec094a1577aeb9765881ab2',
    '4719a4854bc9388b2a866598f9e21c1f14921181',
    '4f052449a40bbc694b11ff99ee481f96b6561759',
    '505561cbebc505dde01873cb5543e16307de1eee',
    '52e438246273b99e55b6292c336293190d9c0e7a',
    '52e6a198a8fb8ed0a31d0910a1edf7590159dc01',
    '53c58d17f5085d207f2a1d7b6b45d627ff3322e3',
    '54a5cd64a2b941183f763b4dc0f20f37ef61ad42',
    '575fdd1004e4ab01c07766e5dd6293427ea4bfde',
    '58b3d50242427ec1afbc000c163715b455affc1c',
    '599dacec76fe8780cafcd4003f41840747f23b50',
    '5a5d0f38887a6d7d5ebad00b9f6e7b599a1fedab',
    '5b32c079cf57c14d7dea6656573d06813f9bdd5d',
    '5bdc19a08eacf9a060f2c0a7a4d50a4adb735094',
    '62d65f57e3c4e45dc19277f893abd0d46e2a7bd0',
    '63a8b4a501ac499e372056a07d499832c830b474',
    '64542402c8d8ca7f5c260c47a75813c799cfd6fd',
    '6692b0cf3f54c708f134055f49d7d814de376b58',
    '69490350d3c11146c7800b4218612d5fff27d7b6',
    '696384f7f1f1e43300280a6ba5948331c4f7cf95',
    '6d13fa9804b58f797f8849784784bc91e9475e69',
    '6e616b842a4ebf0237f91676d082a3543cbd6820',
    '7006fcd90bfa7c3beb715dcf0465913f9e9d10ad',
    '73edc621d1e5a68efe269f46d5bc88d1a1f39f77',
    '7801e7e13dc9a21245487f711e0ec4d54d33b852',
    '7e21585130ae020ac88b04939126f0e2f5cbb827',
    '82bb351f9233db83e167e2ca8390a92f6c84a40a',
    '878a90c774900e961724a563ab75c6e72b448081',
    '8ccc2cbfea1389e14be20ece1ef654cdaced95ea',
    '967bb0e75a899897e278f3a629ed2b43a0bfd17f',
    '98ba5f40b7b9543744e663a96bcdf637c7e2a146',
    '99fd0b4da4ddb6fba21bcb18c924f5e9e8b583e6',
    '9d0e672bedbfa71f3334d271077ef3aa8c18d2b2',
    '9df4a6c003945efd9d91246a370b844205e97662',
    'a188988eddb1efe62067c589ac119342020e4b59',
    'a43ad1cff9ef673c087e1ef3f718371985008ce0',
    'a4a460649ada866e954cbff62b3ffcc9c8297b21',
    'a67fce507af8a0d70a270e8bbe5bcad588431663',
    'a7722a471ca011a94d5fd7774224906001979b78',
    'a879e389c45a448ecddfa29bc9e8497872fe728c',
    'a930025c51bcb9f352e186b40635c71a22cd03d8',
    'aaaa2b5b3e51d6274cb97087c346d69ffa6a8683',
    'ab16f6dea0bbb1c359e644ed3d7cd10f9a7d23d6',
    'ac367dd8c83240b68462e090c8b700b84a5189b1',
    'af9044a1ebe3b14046c9ecbad593d5014904489b',
    'b0150aecfb6fe5bc935a217bee94c37b4ce96038',
    'b02f74736c63d58f6f29cc934027b34962e543cf',
    'b11abe608c43c2ea02e989a0530193ad6de601a1',
    'b1dbb47b8093ebfa066535fa79df48de6bb1206d',
    'b9e6bbb7103237880ef9fb2957084869b06e85b9',
    'ba80c4adec5a354022f9cbd2ceb229a9236dd40c',
    'ba930a6db87fa6527b7216a522a7003bd809e8bb',
    'be9038f083df155ef35819f0e69c2c2ad6c9a516',
    'bf1192e85839b93be04d239fe4fec151e9d548ff',
    'bfd2354484506517cabae105457e17b2b8e32c77',
    'c2eef3d4d4de99110bd3d028c7a88a089cbf9a23',
    'c44156f87f73cf04a7629c32b8db46e9f54c3d2a',
    'c7b23ecb1408be49fb2fa8ef77ffa48640681e34',
    'c7edf5ab57f56bb7302b7d4417496dbc8dfd3924',
    'c891ad48ee63de510e26f8673bed8ebd7de04c7e',
    'ce0f704b0e58f8bdd50fc92864569490bcbd4144',
    'cf4bdbab0351b85cc5173bfba470ab0f30dbd3e2',
    'cf8495732d2bda53a7893bc7f208dff12e7bf9dd',
    'cf9c21590fc6447527a5563f6da3a76f3051ff26',
    'd64a0616bd5bb9c987e43b87bcbf9e89001fbb36',
    'e05827c215102f498e42e04fc0157254a1490f3a',
    'e3473fbbf77fa1261abfc8fd960102c70d3e64bd',
    'e4eb767cd75fb320f302d2ee4f99937bf9936710',
    'e65c18944760294fda5a875ba71e14b27ec0319c',
    'e9c9541581f7bfdfdf340cc823995c6dee3edf20',
    'ea307e511a330877789095b6da52277a07eedd66',
    'eb61c911471867a5fd2466bfd8f29306cea6ebf8',
    'ed29295cd7c32f942a513bb8b0eedcbab15d7407',
    'f23267fe73800d35db382a1919bc28196da5aa8c',
    'f4c7cc694da4766bafa6fec784aa5316027aa538',
    'f5950f3c5552107dcf8d4223383f39811dbed851',
    'f7aad9e212c87829bbe584e65f2419142ff0c3f8',
    'fa63a02628c20e1ab1324ba52cc4497adcbfe7cd',
    'fd22f7efa3cef9988b4036c2d00fc1d3b9c816e8',
    'ffd88098314949e9bf89db3b5b59c1ca3a5b5b9b',
}
FORBIDDEN_BLOBS = WITHHELD_LANDING_BLOBS | WITHHELD_MSA_BLOBS
MARKERS = tuple(name.encode() for name in (
    'fused_minimax_m3_qknorm_rope_kv_insert', 'minimax_m3_index_score', 'minimax_m3_index_topk',
    'minimax_m3_index_decode', 'minimax_m3_sparse_attn', '_gqa_sparse_fwd_kernel', '_gqa_sparse_decode_kernel',
    '_topk_index_kernel', 'sparse_atten_func', 'sparse_decode_atten_func', 'build_k2q_csr',
    'SparseK2qCsrBuilderSm100', 'fmha_sm100', 'MINIMAX_M3_SPARSE', 'MiniMaxM3SparseBackend', 'MiniMaxM3IndexerCache',
))
ROOTS = ('app', 'opt', 'public', 'root', 'tmp', 'usr/local/lib', 'var/cache', 'hf')
FORBIDDEN_DISTS = ('fmha-sm100', 'fmha_sm100', 'msa', 'kernels', 'tensorrt-llm', 'tensorrt_llm')
PINNED_DISTS = {'transformers': '5.5.3', 'flashinfer-python': '0.6.12'}


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
    """gzip, or a joblib-style zlib stream (with or without a ZF length header); unknown formats fail."""
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


#: Base material that names the marker without being the withheld package, exempt by path AND
#: content hash only (a changed file is scanned as any other). FlashMLA at a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b,
#: fetched by the base's own cmake/external_projects/flashmla.cmake; its lines 40 and 181 name FlashMLA's own
#: tests/test_fmha_sm100.py.
EXEMPT = {
    'app/.deps/flashmla-src/README.md': '46c001c53800f47148918f4fc2969102492dd5164f710fc015db1622f6f9929a',
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def filesystem(root):
    hits, scanned, errors, exempt = [], 0, [], []
    tensor_blobs = {path.resolve() for path in (root / 'hf').glob('**/snapshots/*/*.safetensors')}
    for name in ROOTS:
        base = root / name
        if not base.exists(): continue
        for path in base.rglob('*'):
            if path.is_symlink() or not path.is_file() or '.git' in path.parts:
                continue
            if path.suffix == '.safetensors' or path in tensor_blobs: continue
            rel = str(path.relative_to(root))
            if rel in EXEMPT and _sha256(path) == EXEMPT[rel]:
                exempt.append({'path': str(path), 'sha256': EXEMPT[rel], 'reason': 'public upstream file pinned by sha256'})
                continue
            scanned += 1
            try:
                if path.suffix in ('.whl', '.zip'):
                    with zipfile.ZipFile(path) as archive:
                        for member in archive.infolist():
                            if not member.is_dir():
                                with archive.open(member) as file:
                                    inspect_stream(file, str(path) + '!' + member.filename, hits)
                elif any(path.name.endswith(s) for s in ('.tar', '.tar.gz', '.tgz', '.tar.xz', '.tar.bz2')):
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
    deps = root / 'app' / '.deps'
    fetched = sorted(str(p) for p in deps.glob('*fmha_sm100*')) if deps.exists() else []
    return {'files': scanned, 'hits': hits, 'unreadable': errors, 'deps_fmha_sm100': fetched, 'exempt': exempt}


def git_audit(root):
    app = root / 'app'
    def git(*args):
        return subprocess.check_output(['git', '-C', str(app), *args], text=True)
    rows = git('cat-file', '--batch-all-objects', '--batch-check=%(objectname) %(objecttype)').splitlines()
    forbidden = [row for row in rows if row.split()[0] in FORBIDDEN_BLOBS]
    blob_ids = [row.split()[0] for row in rows if row.endswith(' blob')]
    marker_hits = []
    proc = subprocess.Popen(['git', '-C', str(app), 'cat-file', '--batch'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    def feed():  # from its own thread: written whole before reading, both pipes fill and the two processes deadlock
        proc.stdin.write(('\n'.join(blob_ids) + '\n').encode())
        proc.stdin.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    for _ in blob_ids:
        header = proc.stdout.readline().split()
        size = int(header[2])
        body = proc.stdout.read(size)
        proc.stdout.read(1)
        for marker in MARKERS:
            if marker in body:
                marker_hits.append({'blob': header[0].decode(), 'marker': marker.decode()})
    feeder.join()
    proc.wait()
    refs = git('for-each-ref', '--format=%(refname)').splitlines()
    non_ancestral = [ref for ref in refs if subprocess.run(
        ['git', '-C', str(app), 'merge-base', '--is-ancestor', ref, 'HEAD'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode]
    unreachable = git('fsck', '--full', '--no-reflogs', '--unreachable').strip()
    alternates = (app / '.git/objects/info/alternates').exists()
    return {'objects': len(rows), 'blobs_scanned': len(blob_ids), 'forbidden': forbidden,
            'marker_hits': marker_hits, 'refs': refs, 'non_ancestral': non_ancestral,
            'unreachable': unreachable, 'alternates': alternates}


def stores(root):
    """Every Git object store in the image except /app's own: any directory holding a HEAD file and an objects/
    directory (a .git directory, a submodule store under .git/modules, a bare clone such as cargo's git database)."""
    skip = {root / name for name in ("proc", "sys", "dev")}
    own = root / 'app' / '.git'
    found = []
    for top, dirs, files in os.walk(root, topdown=True, followlinks=False):
        here = Path(top)
        if here in skip:
            dirs[:] = []
            continue
        if 'HEAD' in files and 'objects' in dirs:
            if here != own:
                found.append(here)
            dirs.remove('objects')  # a store's objects are read through git, not walked
    return sorted(found)


def store_audit(store):
    def git(*args):
        return subprocess.run(['git', '--git-dir', str(store), *args], capture_output=True, text=True).stdout
    rows = [line.split() for line in git('cat-file', '--batch-all-objects', '--unordered',
                                         '--batch-check=%(objectname) %(objecttype)').splitlines() if line.strip()]
    forbidden = [oid for oid, _ in rows if oid in FORBIDDEN_BLOBS or oid in FORBIDDEN_IDENTITIES]
    blob_ids = [oid for oid, kind in rows if kind == 'blob']
    exempt_content = set(EXEMPT.values())
    marker_hits, exempt = [], []
    proc = subprocess.Popen(['git', '--git-dir', str(store), 'cat-file', '--batch'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    def feed():  # from its own thread, as git_audit does, so the two pipes cannot deadlock
        proc.stdin.write(('\n'.join(blob_ids) + '\n').encode())
        proc.stdin.close()

    feeder = threading.Thread(target=feed, daemon=True)
    feeder.start()
    for _ in blob_ids:
        header = proc.stdout.readline().split()
        body = proc.stdout.read(int(header[2]))
        proc.stdout.read(1)
        hits = [marker.decode() for marker in MARKERS if marker in body]
        if hits and hashlib.sha256(body).hexdigest() in exempt_content:
            exempt.append({'blob': header[0].decode(), 'reason': 'public upstream file pinned by sha256'})
        elif hits:
            marker_hits.append({'blob': header[0].decode(), 'markers': hits})
    feeder.join()
    proc.wait()
    head = git('rev-parse', '--verify', '-q', 'HEAD^{commit}').strip() or None
    refs = [line.split() for line in git('for-each-ref', '--format=%(refname) %(objectname) %(*objectname)').splitlines()
            if line.strip()]
    points = {ref[0]: ref[2] if len(ref) > 2 else ref[1] for ref in refs}
    if head:
        not_at_head = sorted(name for name, target in points.items() if target != head)
    else:
        not_at_head = sorted(points) if len(set(points.values())) > 1 else []
    return {'store': str(store), 'objects': len(rows), 'blobs_scanned': len(blob_ids), 'head': head,
            'refs': sorted(points), 'forbidden': forbidden, 'marker_hits': marker_hits, 'exempt': exempt,
            'refs_not_at_head': not_at_head}


def environments():
    from importlib.metadata import distributions
    report = {}
    for interpreter in ('/opt/vllm/bin/python', '/opt/infb/tools/bin/python'):
        if not Path(interpreter).exists():
            continue
        out = subprocess.run([interpreter, '-c', (
            'import json; from importlib.metadata import distributions; '
            'print(json.dumps(sorted({(d.metadata["Name"] or "").lower().replace("_", "-"): d.version '
            'for d in distributions()}.items())))')], capture_output=True, text=True, check=True, cwd='/').stdout
        dists = dict(json.loads(out))
        report[interpreter] = {
            'forbidden': sorted(n for n in dists if n in FORBIDDEN_DISTS or n.startswith('tensorrt')),
            'pinned': {n: dists.get(n) for n in PINNED_DISTS},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/'))
    parser.add_argument('--filesystem-only', action='store_true')
    args = parser.parse_args()
    report = filesystem(args.root)
    failed = bool(report['hits'] or report['unreadable'] or report['deps_fmha_sm100'])
    if not args.filesystem_only:
        report['git'] = git_audit(args.root)
        failed |= any(report['git'][key] for key in ('forbidden', 'marker_hits', 'non_ancestral', 'unreachable',
                                                       'alternates'))
        report['stores'] = [store_audit(store) for store in stores(args.root)]
        failed |= any(st['forbidden'] or st['marker_hits'] or st['refs_not_at_head'] for st in report['stores'])
        report['environments'] = environments()
        vllm_env = report['environments'].get('/opt/vllm/bin/python', {})
        failed |= bool(vllm_env.get('forbidden')) or vllm_env.get('pinned') != PINNED_DISTS
        failed |= any(e['forbidden'] for e in report['environments'].values())
    report['passed'] = not failed
    print(json.dumps(report, indent=1))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
