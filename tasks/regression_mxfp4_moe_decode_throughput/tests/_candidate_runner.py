#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Decode-throughput bench runner for regression_mxfp4_moe_decode_throughput.

Per-task runner, invoked as a subprocess by `tests/_bench_rounds.py` for BOTH the pristine-baseline
tree and the candidate tree. The verifier reads the two result JSONs and computes the scored ratio.

WHY THIS ONE USES THE STOCK BENCH, where the sibling SGLang task in this family hand-rolls its own.
That task's defect is only visible against three ordered phases with shared prefixes, which no stock
bench produces, so it had to build the workload itself. This defect is a per-decode-step cost in the
expert matmul: it needs decode steps and a small batch, and `sglang.bench_serving --dataset-name
random` produces exactly that. Writing a second load generator here would add a measurement layer of
my own to get wrong, and its output keys would then need a normaliser that the stock ones already
have. `bench_serving --output-file` emits `output_throughput`, `mean_tpot_ms`, `request_throughput`
and friends, which `_verifier_lib/metrics.py` already maps into the standard record.

SCORED: `output_throughput`, tokens/s, HIGHER IS BETTER, so the verifier does not invert the ratio.
Recorded and not scored: `mean_tpot_ms`, which moves by the same factor and is the cross-check -- a
per-step cost shows up in both, a per-request cost would move only one of them.

THE CONCURRENCY IS THE EXPERIMENT. 32 experts at top-k 4 put roughly B*4/32 rows on each expert, so
the regime this defect lives in survives only while the batch is small: measured 1.49x at concurrency
4, 1.10x at 16, and 1.00x from 24 up. A saturating burst reads 1.0 with nothing fixed. This is the
mirror image of the sibling MoE task in this family, where concurrency 32 is required and batch size
1 certifies nothing -- do not copy a shape between them.

Writes `sglang_bench_serving.json` + `vllm_bench_manifest.json` into `--raw-dir`. The manifest
basename is the family's, not a typo -- `verify_regression_real.py` looks for exactly that name.

AND `correctness_probe.json`, since 2026-09-24: after the scored bench, against the same live server,
a fixed prompt set sent one request at a time -- the teacher-forced log-probabilities the tree assigns
to each prompt, plus a short greedy continuation. It runs in BOTH arms (this runner is what benches
both), and `verify_regression_real.check_outputs_agree` compares the two. See `_correctness_probe`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_lib.bench_stats import load_matrix_spec, point_raw_dir  # noqa: E402

CONFIG = {
    "tp_size": 1,
    "port": 30137,
    "server_ready_timeout_sec": 900.0,
    "warmup_prompts": 8,
}

REQUIRED_BANNER = "enabling triton_kernels MOE kernel"


def _say(msg: str) -> None:
    print(f"[decode_bench] {msg}", flush=True)


def _workspace() -> Path:
    w = os.environ.get("INFB_WORKSPACE")
    return Path(w) if w and Path(w).is_dir() else Path("/app")


def _venv_python() -> str:
    v = os.environ.get("INFB_VLLM_PYTHON")
    if v and Path(v).is_file():
        return v
    venv = os.environ.get("INFB_VENV")
    if venv and (Path(venv) / "bin" / "python").is_file():
        return str(Path(venv) / "bin" / "python")
    return sys.executable


def _model(ws: Path) -> str:
    v = os.environ.get("INFB_MODEL_PATH")
    if v and Path(v).exists():
        return v
    link = ws / "model"
    if link.exists():
        return str(link)
    raise SystemExit("model not resolved (set INFB_MODEL_PATH or <workspace>/model)")


def _select_extra_points(selection: str, spec: dict) -> list[str]:
    """The non-anchor points this invocation benches, scored ones first, then guards."""
    if selection in ("", "anchor"):
        return []
    candidates = [n for n in spec["scored"] if n != spec["anchor"]] + list(spec["guards"])
    if selection != "all":
        wanted = [s.strip() for s in selection.split(",") if s.strip()]
        unknown = sorted(set(wanted) - set(spec["points"]))
        if unknown:
            raise SystemExit(f"--workloads names points the spec does not declare: {unknown}")
        candidates = [n for n in candidates if n in wanted]
    return [n for n in candidates if not isinstance(spec["points"][n].get("from"), str)]


def _bench_point(python: str, port: int, env: dict, name: str, wl: dict, pdir: Path,
                 serve_cmd: list[str], server_log: Path) -> dict:
    """One extra point against the already-running server; its own record and manifest under `pdir`."""
    pdir.mkdir(parents=True, exist_ok=True)
    jsonl = pdir / "bench_serving_scored.jsonl"
    jsonl.unlink(missing_ok=True)
    cmd = _bench_argv(python, port, wl, int(wl["num_prompts"]), jsonl)
    _say(f"point {name}: {shlex.join(cmd)}")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, check=False).returncode
    error = None
    if rc != 0:
        error = f"bench exited rc={rc}"
    else:
        try:
            rec = _read_bench_record(jsonl)
            if not rec.get("output_throughput"):
                error = f"no output_throughput in the record; keys {sorted(rec)}"
            elif int(rec.get("completed", 0)) != int(wl["num_prompts"]):
                error = f"completed {rec.get('completed')} of {wl['num_prompts']} requests"
            else:
                result = dict(rec)
                result["workload"] = wl
                (pdir / "sglang_bench_serving.json").write_text(json.dumps(result, indent=2) + "\n")
                _say(f"point {name}: output throughput {rec['output_throughput']:.2f} tok/s   "
                     f"mean TPOT {rec.get('mean_tpot_ms', float('nan')):.3f} ms   "
                     f"mean TTFT {rec.get('mean_ttft_ms', float('nan')):.1f} ms")
        except SystemExit as exc:
            error = str(exc)
    if error:
        _say(f"point {name}: FAILED -- {error}")
    (pdir / "vllm_bench_manifest.json").write_text(json.dumps({
        "mode": "serve", "returncode": rc if rc else (1 if error else 0), "error": error,
        "point": name, "workload": wl,
        "command": f"sglang bench_serving (vllm bench manifest shape): {shlex.join(cmd)}",
        "serve_command": shlex.join(serve_cmd), "bench_command": shlex.join(cmd),
        "elapsed_sec": round(time.time() - t0, 2),
    }, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "returncode": rc, "error": error}


def _primary_workload() -> dict:
    """The TRUSTED copy, resolved from this file's own directory."""
    doc = json.loads((Path(__file__).resolve().parent / "workloads.json").read_text())
    wls = doc["workloads"]
    name = doc.get("primary_workload")
    for w in wls:
        if w.get("name") == name:
            return w
    return wls[0]


def _pdeathsig_preexec() -> None:
    """Ask the kernel to SIGKILL the server if this runner dies."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001 -- best effort; explicit teardown still runs
        pass
    os.setpgrp()


def _port_busy(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 -- fixed localhost
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def _wait_ready(proc: subprocess.Popen, url: str, timeout: float, log: Path) -> float:
    """Block until the server answers, or die with the server's own tail."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(
                f"server exited rc={proc.returncode} before becoming ready\n"
                f"--- server log ---\n{_tail(log)}"
            )
        if _http_ok(url):
            return time.time() - t0
        time.sleep(1.0)
    raise SystemExit(f"server not ready after {timeout:.0f}s\n--- server log ---\n{_tail(log)}")


NUMERIC_PROBE = {"seed": 20260919, "n_tokens": 192, "file": "moe_numeric_probe.json"}


def _vocab_size(model: str) -> int | None:
    try:
        cfg = json.loads((Path(model) / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("vocab_size",):
        v = cfg.get(key)
        if isinstance(v, int) and v > 1024:
            return v
    txt = cfg.get("text_config")
    if isinstance(txt, dict) and isinstance(txt.get("vocab_size"), int):
        return txt["vocab_size"]
    return None


def _numeric_fingerprint(base_url: str, model: str) -> dict:
    """Teacher-forced input logprobs on a FIXED token sequence, ONE request, no generation."""
    out: dict = {"available": False}
    try:
        n = int(NUMERIC_PROBE["n_tokens"])
        vocab = _vocab_size(model)
        rng = random.Random(NUMERIC_PROBE["seed"])
        if vocab:
            ids = [rng.randrange(16, max(32, vocab - 16)) for _ in range(n)]
            body: dict = {"input_ids": ids}
            out["prompt_kind"] = "token_ids"
            out["prompt_sha256"] = hashlib.sha256(
                ",".join(str(i) for i in ids).encode()).hexdigest()[:16]
        else:
            text = " ".join(f"tok{rng.randrange(0, 99999)}" for _ in range(n))
            body = {"text": text}
            out["prompt_kind"] = "text_fallback"
            out["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()[:16]
        body.update({
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 1,
        })
        req = urllib.request.Request(
            f"{base_url}/generate", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode())
        meta = (payload[0] if isinstance(payload, list) else payload).get("meta_info") or {}
        itl = meta.get("input_token_logprobs")
        if not isinstance(itl, list) or not itl:
            out["reason"] = f"server returned no input_token_logprobs; meta keys {sorted(meta)}"
            return out
        lps = [float(e[0]) for e in itl if isinstance(e, (list, tuple)) and e and e[0] is not None]
        tops = meta.get("input_top_logprobs")
        top1: list[int] = []
        if isinstance(tops, list):
            for pos in tops:
                if isinstance(pos, list) and pos and isinstance(pos[0], (list, tuple)) and len(pos[0]) > 1:
                    top1.append(int(pos[0][1]))
        out.update({
            "available": True,
            "n_positions": len(lps),
            "logprobs": [round(v, 6) for v in lps],
            "sum_logprob": round(sum(lps), 6),
            "top1_ids": top1,
            "n_top1": len(top1),
        })
    except Exception as exc:  # noqa: BLE001 -- a broken probe is "no reading", never a crashed arm
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out


def _serve_argv(python: str, model: str, port: int) -> list[str]:
    return [
        python, "-m", "sglang.launch_server",
        "--model-path", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", str(CONFIG["tp_size"]),
        "--log-level", "info",
    ]


def _bench_argv(python: str, port: int, wl: dict, n_prompts: int, out_file: Path) -> list[str]:
    declared_ratio = float(wl.get("range_ratio", 1.0))
    if declared_ratio != 1.0:
        raise SystemExit(f"workload declares range_ratio={declared_ratio}; this runner pins 1.0 (exact lengths)")
    return [
        python, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dataset-name", "random",
        "--dataset-path", "/opt/infb/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
        "--random-input-len", str(int(wl["input_len"])),
        "--random-output-len", str(int(wl["output_len"])),
        "--random-range-ratio", "1.0",
        "--num-prompts", str(n_prompts),
        "--max-concurrency", str(int(wl["max_concurrency"])),
        "--output-file", str(out_file),
    ]


def _read_bench_record(path: Path) -> dict:
    """`bench_serving --output-file` appends ONE JSON object per line. Take the last."""
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty -- bench_serving wrote no record")
    return json.loads(lines[-1])


def _assert_path_taken(log: Path) -> None:
    """The engine must have taken the MoE path this task measures."""
    text = log.read_text(errors="replace")
    if REQUIRED_BANNER not in text:
        raise SystemExit(
            "the server did not report taking the MoE path this task measures "
            f"(looked for {REQUIRED_BANNER!r} in the startup log).\n"
            "That path is gated on the vendored `triton_kernels` package being importable; without "
            "it the engine serves this model correctly on another backend and BOTH ARMS READ 1.00x. "
            "This is an environment fault, not a slow candidate.\n"
            f"--- server log ---\n{_tail(log, 60)}"
        )


CORRECTNESS_PROBE_FILE = "correctness_probe.json"
CORRECTNESS_MAX_NEW_TOKENS = 32
CORRECTNESS_TOP_LOGPROBS = 5
CORRECTNESS_REQUEST_TIMEOUT_SEC = 120.0
CORRECTNESS_BUDGET_SEC = 300.0
CORRECTNESS_PROMPTS: tuple[str, ...] = (
    "The water cycle describes how water moves between the oceans, the atmosphere and the land. "
    "Sunlight warms the surface of seas and lakes, and some of that water evaporates into invisible "
    "vapour. Plants add more vapour through their leaves, a process called transpiration. As the moist "
    "air rises it cools, and the vapour condenses around tiny particles of dust to form the droplets "
    "that make up clouds. When the droplets grow heavy enough they fall as rain, snow or hail. Some of "
    "that precipitation soaks into the ground and refills underground aquifers, some runs off into "
    "streams and rivers, and some is stored for months or years as ice and snow. Eventually the water "
    "returns to the sea, and the cycle begins again. Over the course of a single year, the water that "
    "evaporates from the oceans",
    "def merge_sorted(left, right):\n"
    "    \"\"\"Merge two sorted lists into a single sorted list.\n\n"
    "    Both inputs must already be sorted in ascending order. The function walks the two lists with\n"
    "    one index each and always appends the smaller head element, so it runs in linear time.\n"
    "    \"\"\"\n"
    "    result = []\n"
    "    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n"
    "            result.append(left[i])\n"
    "            i += 1\n"
    "        else:\n"
    "            result.append(right[j])\n"
    "            j += 1\n"
    "    result.extend(left[i:])\n"
    "    result.extend(right[j:])\n"
    "    return result\n\n\n"
    "def merge_sort(items):\n"
    "    \"\"\"Sort a list with the merge sort algorithm and return a new list.\"\"\"\n"
    "    if len(items) <= 1:\n"
    "        return list(items)\n",
    "How to bake a simple loaf of bread\n\n"
    "Ingredients: 500 grams of strong white flour, 10 grams of salt, 7 grams of dried yeast, 30 "
    "millilitres of olive oil and 300 millilitres of lukewarm water.\n\n"
    "1. Put the flour in a large bowl and add the salt to one side and the yeast to the other.\n"
    "2. Pour in the oil and three quarters of the water, then mix by hand, adding the rest of the "
    "water a little at a time until you have a soft dough.\n"
    "3. Tip the dough onto a lightly oiled surface and knead it for about ten minutes, until it feels "
    "smooth and stretchy.\n"
    "4. Return it to the bowl, cover it with a clean tea towel and leave it somewhere warm for an "
    "hour, or until it has doubled in size.\n"
    "5.",
    "Before movable type, books in Europe were copied by hand, usually by scribes working in "
    "monasteries. A single copy of a long work could take months to produce, so books were rare and "
    "expensive, and most people never owned one. In the middle of the fifteenth century, a craftsman "
    "in the German city of Mainz developed a system of individual metal letters that could be arranged "
    "into lines, inked and pressed onto paper, then taken apart and used again. Combined with an "
    "oil-based ink and a press adapted from the kind used to make wine, the system made it possible to "
    "print hundreds of identical pages in a day. Within a few decades printing workshops had opened in "
    "cities across the continent, and the price of books fell sharply. The consequences reached far "
    "beyond",
    "-- Schema for a small library database\n"
    "CREATE TABLE authors (\n"
    "    author_id   INTEGER PRIMARY KEY,\n"
    "    name        TEXT NOT NULL,\n"
    "    birth_year  INTEGER\n"
    ");\n\n"
    "CREATE TABLE books (\n"
    "    book_id     INTEGER PRIMARY KEY,\n"
    "    title       TEXT NOT NULL,\n"
    "    author_id   INTEGER NOT NULL REFERENCES authors(author_id),\n"
    "    published   INTEGER,\n"
    "    copies      INTEGER DEFAULT 1\n"
    ");\n\n"
    "CREATE TABLE loans (\n"
    "    loan_id     INTEGER PRIMARY KEY,\n"
    "    book_id     INTEGER NOT NULL REFERENCES books(book_id),\n"
    "    member_id   INTEGER NOT NULL,\n"
    "    loaned_on   DATE NOT NULL,\n"
    "    returned_on DATE\n"
    ");\n\n"
    "-- Titles that are currently on loan, with the name of their author,\n"
    "-- ordered by the date they were borrowed.\n"
    "SELECT",
    "Problem: A train leaves a station at 9:00 in the morning travelling at 80 kilometres per hour. A "
    "second train leaves the same station at 10:30 on the same track, travelling at 120 kilometres per "
    "hour. At what time does the second train catch up with the first?\n\n"
    "Solution: By 10:30 the first train has been travelling for one and a half hours, so it is "
    "80 x 1.5 = 120 kilometres ahead. The second train closes that gap at 120 - 80 = 40 kilometres per "
    "hour, and closing a gap of 120 kilometres at 40 kilometres per hour takes 120 / 40 = 3 hours. The "
    "second train therefore catches the first at 10:30 + 3 hours = 13:30, which is half past one in "
    "the afternoon. At that moment both trains are",
    "Subject: Moving the quarterly review to Thursday\n\n"
    "Hi everyone,\n\n"
    "Because the finance team needs two more days to close the accounts for the quarter, we are moving "
    "the quarterly review from Tuesday morning to Thursday afternoon. The meeting will now start at 2 "
    "pm in the large conference room on the third floor and should finish by 4 pm. Please bring the "
    "updated figures for your department, and send any slides to me by Wednesday evening so that I can "
    "combine them into a single deck. If you cannot attend in person, a video link will be included in "
    "the calendar invitation. Sorry for the late change, and thank you for your flexibility.\n\n"
    "Best regards,\nMaria\n\n"
    "Reply:\n\nHi Maria,\n\nThanks for letting us know.",
    "{\n"
    "  \"service\": \"inventory-api\",\n"
    "  \"version\": \"2.4.1\",\n"
    "  \"listen\": {\"host\": \"0.0.0.0\", \"port\": 8080},\n"
    "  \"database\": {\n"
    "    \"driver\": \"postgres\",\n"
    "    \"host\": \"db.internal\",\n"
    "    \"port\": 5432,\n"
    "    \"name\": \"inventory\",\n"
    "    \"pool\": {\"min_connections\": 2, \"max_connections\": 20, \"idle_timeout_seconds\": 300}\n"
    "  },\n"
    "  \"cache\": {\"enabled\": true, \"backend\": \"redis\", \"ttl_seconds\": 60},\n"
    "  \"logging\": {\"level\": \"info\", \"format\": \"json\", \"destination\": \"stdout\"},\n"
    "  \"features\": {\n"
    "    \"bulk_import\": true,\n"
    "    \"audit_trail\": true,\n"
    "    \"experimental_search\": false\n"
    "  },\n"
    "  \"rate_limits\": [\n"
    "    {\"route\": \"/items\", \"requests_per_minute\": 600},\n"
    "    {\"route\": \"/orders\", \"requests_per_minute\": 120},\n",
    "A CPU cache is a small, fast memory that sits between the processor cores and the much larger but "
    "slower main memory. When a core needs a value it first checks the level one cache, which is "
    "private to that core and can answer in a few cycles. If the value is not there, the request moves "
    "on to the level two cache and then to a shared level three cache, each larger and slower than the "
    "last. Only if every level misses does the request go out to main memory, which can take hundreds "
    "of cycles. Caches work because programs tend to reuse the data they touched recently and to access "
    "memory locations that are close together, so data is moved in fixed-size blocks called cache "
    "lines, typically sixty-four bytes long. When a cache is full and a new line must be brought in, "
    "the hardware has to choose",
    "Customer: Hello, I bought a pair of running shoes here last week, but they are half a size too "
    "small. Can I exchange them?\n"
    "Assistant: Of course. Do you have the receipt with you?\n"
    "Customer: Yes, here it is. I have only worn them once, indoors, to check the fit.\n"
    "Assistant: That is fine, they look brand new. Let me see whether we have the same model in a "
    "larger size. What size are these?\n"
    "Customer: They are a nine, so I would like a nine and a half.\n"
    "Assistant: We have a nine and a half in the same colour, and also in blue if you would prefer "
    "that.\n"
    "Customer: The same colour is fine, thank you.\n"
    "Assistant: Here you are. Would you like to try them on before we complete the exchange?\n"
    "Customer:",
    "A list of countries and their capital cities:\n\n"
    "France - Paris\nGermany - Berlin\nItaly - Rome\nSpain - Madrid\nPortugal - Lisbon\n"
    "Netherlands - Amsterdam\nBelgium - Brussels\nAustria - Vienna\nSwitzerland - Bern\n"
    "Poland - Warsaw\nSweden - Stockholm\nNorway - Oslo\nDenmark - Copenhagen\nFinland - Helsinki\n"
    "Ireland - Dublin\nGreece - Athens\nHungary - Budapest\nCzech Republic - Prague\nJapan - Tokyo\n"
    "China - Beijing\nIndia - New Delhi\nEgypt - Cairo\nKenya - Nairobi\nCanada - Ottawa\n"
    "Mexico - Mexico City\nBrazil - Brasilia\nArgentina -",
    "I have used this electric kettle every day for three months, and overall I am happy with it. It "
    "boils a full litre of water in a little over two minutes, which is noticeably faster than the old "
    "stovetop kettle it replaced. The handle stays cool, the lid opens with a single button, and the "
    "automatic switch-off has worked every time. The water window on the side is easy to read, even "
    "from across the kitchen. My complaints are small ones. The power cord is rather short, so the "
    "kettle has to sit close to the socket, and the base is slightly wider than I expected. It is also "
    "louder than some models, with a noticeable rumble just before it reaches the boil. For the price, "
    "though, I would still recommend it to anyone who",
    "#!/usr/bin/env bash\n"
    "# Back up a directory into a dated, compressed archive and keep only the newest seven.\n"
    "set -euo pipefail\n\n"
    "source_dir=\"${1:?usage: backup.sh <source-dir> <backup-dir>}\"\n"
    "backup_dir=\"${2:?usage: backup.sh <source-dir> <backup-dir>}\"\n"
    "stamp=\"$(date +%Y-%m-%d_%H%M%S)\"\n"
    "archive=\"${backup_dir}/backup_${stamp}.tar.gz\"\n\n"
    "mkdir -p \"${backup_dir}\"\n"
    "echo \"Creating ${archive} from ${source_dir}\"\n"
    "tar -czf \"${archive}\" -C \"$(dirname \"${source_dir}\")\" \"$(basename \"${source_dir}\")\"\n\n"
    "# Delete everything except the seven most recent archives.\n"
    "ls -1t \"${backup_dir}\"/backup_*.tar.gz | tail -n +8 | while read -r old; do\n",
    "Photosynthesis is the process by which green plants, algae and some bacteria turn light into "
    "chemical energy. It takes place mainly in the leaves, inside small structures called "
    "chloroplasts, which contain the green pigment chlorophyll. Chlorophyll absorbs red and blue light "
    "strongly and reflects green light, which is why leaves look green to us. In the first stage of the "
    "process, the absorbed energy is used to split water molecules, releasing oxygen as a by-product "
    "and producing molecules that carry energy. In the second stage, those molecules are used to take "
    "carbon dioxide from the air and build it into sugars. The overall reaction can be summarised as "
    "six molecules of carbon dioxide plus six molecules of water, in the presence of light, producing "
    "one molecule of glucose and",
    "Le petit village se trouve au bord d'une rivière, entouré de collines couvertes de vignes. Chaque "
    "matin, le boulanger ouvre sa boutique avant le lever du soleil, et l'odeur du pain chaud se répand "
    "dans les rues étroites. Les enfants traversent la place pour aller à l'école, tandis que les "
    "anciens s'installent sur les bancs près de la fontaine pour discuter des nouvelles du jour. Le "
    "samedi, un marché s'installe devant la mairie : on y trouve des fromages, des légumes frais, du "
    "miel et des fleurs. En automne, pendant les vendanges, tout le village participe à la récolte du "
    "raisin, et le soir on se retrouve pour un grand repas partagé. Les visiteurs qui passent par ici "
    "disent souvent que",
    "Setting up a game of chess\n\n"
    "Place the board so that each player has a light square in the corner on their right-hand side. "
    "The pieces start on the two rows closest to each player. On the second row, place a line of eight "
    "pawns. On the first row, the rooks go in the two corners, the knights stand next to the rooks, and "
    "the bishops stand next to the knights. The queen goes on the remaining square of her own colour, "
    "so the white queen starts on a light square and the black queen on a dark one. The king takes the "
    "last empty square, beside the queen. When both sides are set up, the two kings face each other "
    "along the same file, and so do the two queens. White always moves first, and the players then "
    "take turns. The object of the game is to",
)


def _sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in ids).encode("ascii")).hexdigest()


def _logprob_entries(raw: object) -> tuple[list[int], list[float | None]]:
    """(token ids, logprobs) from a `meta_info.*_token_logprobs` list of (logprob, token_id, text)."""
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of (logprob, token_id, text), got {type(raw).__name__}")
    ids: list[int] = []
    lps: list[float | None] = []
    for entry in raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or entry[1] is None:
            raise ValueError(f"malformed logprob entry {entry!r:.80}")
        ids.append(int(entry[1]))
        lps.append(round(float(entry[0]), 6) if entry[0] is not None else None)
    return ids, lps


def _top_entries(raw: object) -> list[list[list[float]] | None]:
    """Per position, [[token_id, logprob], ...] from `meta_info.*_top_logprobs`, None where absent."""
    if not isinstance(raw, list):
        return []
    out: list[list[list[float]] | None] = []
    for position in raw:
        if not isinstance(position, (list, tuple)) or not position:
            out.append(None)
            continue
        out.append([[int(e[1]), round(float(e[0]), 6)] for e in position
                    if isinstance(e, (list, tuple)) and len(e) >= 2 and e[0] is not None
                    and e[1] is not None])
    return out


def _correctness_probe(base_url: str) -> dict:
    """The fixed prompts, one request at a time, against the live server. See the block above."""
    started = time.time()
    probe: dict = {
        "schema": 1,
        "comparison": "teacher_forced_prompt_logprobs",
        "concurrency": 1,
        "n_prompts": len(CORRECTNESS_PROMPTS),
        "prompt_set_sha256": hashlib.sha256(
            "\x00".join(CORRECTNESS_PROMPTS).encode("utf-8")).hexdigest(),
        "max_new_tokens": CORRECTNESS_MAX_NEW_TOKENS,
        "top_logprobs_num": CORRECTNESS_TOP_LOGPROBS,
        "prompts": [],
        "error": None,
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for text in CORRECTNESS_PROMPTS:
            if time.time() - started > CORRECTNESS_BUDGET_SEC:
                raise TimeoutError(f"probe budget {CORRECTNESS_BUDGET_SEC:.0f}s exceeded after "
                                   f"{len(probe['prompts'])} prompts")
            body = {
                "text": text,
                "sampling_params": {"temperature": 0.0,
                                    "max_new_tokens": CORRECTNESS_MAX_NEW_TOKENS,
                                    "ignore_eos": True},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": CORRECTNESS_TOP_LOGPROBS,
            }
            req = urllib.request.Request(f"{base_url}/generate", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=CORRECTNESS_REQUEST_TIMEOUT_SEC) as fh:  # noqa: S310
                payload = json.loads(fh.read().decode())
            item = payload[0] if isinstance(payload, list) else payload
            meta = (item or {}).get("meta_info") if isinstance(item, dict) else None
            if not isinstance(meta, dict):
                raise ValueError(f"response carries no meta_info: {str(payload)[:200]}")
            prompt_ids, prompt_lps = _logprob_entries(meta.get("input_token_logprobs"))
            output_ids, output_lps = _logprob_entries(meta.get("output_token_logprobs"))
            if not prompt_ids or not output_ids:
                raise ValueError(f"no token logprobs (meta_info keys {sorted(meta)})")
            prompt_top = _top_entries(meta.get("input_top_logprobs"))
            probe["prompts"].append({
                "n_prompt_tokens": len(prompt_ids),
                "prompt_ids_sha256": _sha256_ids(prompt_ids),
                "prompt_logprobs": prompt_lps,
                "prompt_top1": [p[0][0] if p else None for p in prompt_top],
                "output_ids": output_ids,
                "output_logprobs": output_lps,
                "output_top": _top_entries(meta.get("output_top_logprobs")),
                "output_digest": _sha256_ids(output_ids),
                "output_text": str(item.get("text", ""))[:400],
            })
    except Exception as exc:  # noqa: BLE001 -- recorded as unmeasured, never silently "passing"
        probe["error"] = f"{type(exc).__name__}: {exc}"[:500]
    probe["elapsed_sec"] = round(time.time() - started, 2)
    probe["output_digest"] = hashlib.sha256(
        "\n".join(p["output_digest"] for p in probe["prompts"]).encode("ascii")).hexdigest()
    return probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve"])
    ap.add_argument("--repo", required=True, help="engine source tree to overlay via PYTHONPATH")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument(
        "--workloads", default="anchor",
        help="`anchor` (the primary workload only, the warmup default), `all` (every scored and guard "
             "point of tests/workloads.json in one server session), or a comma-separated list",
    )
    args = ap.parse_args()

    ws = _workspace()
    repo = Path(args.repo).resolve()
    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    out_json = raw / "sglang_bench_serving.json"
    manifest = raw / "vllm_bench_manifest.json"
    server_log = raw / "sglang_server.log"
    bench_jsonl = raw / "bench_serving_scored.jsonl"
    warm_jsonl = raw / "bench_serving_warmup.jsonl"
    probe_json = raw / CORRECTNESS_PROBE_FILE
    for stale in (bench_jsonl, warm_jsonl, probe_json):
        stale.unlink(missing_ok=True)

    python = _venv_python()
    model = _model(ws)
    wl = _primary_workload()
    spec = load_matrix_spec(Path(__file__).resolve().parent / "workloads.json")
    extra_points = _select_extra_points(args.workloads, spec)
    port = int(os.environ.get("INFB_SERVE_PORT") or CONFIG["port"])
    base_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}/python:{env.get('PYTHONPATH', '')}".rstrip(":")

    if _port_busy(port):
        raise SystemExit(f"port {port} already in use before this arm started; refusing to bench "
                         f"against someone else's server")

    serve_cmd = _serve_argv(python, model, port)
    _say(f"serve: {shlex.join(serve_cmd)}")

    t0 = time.time()
    rc = 1
    result: dict = {}
    bench_cmd: list[str] = []
    points_benched: list[dict] = []
    with server_log.open("w") as slog:
        proc = subprocess.Popen(serve_cmd, env=env, stdout=slog, stderr=subprocess.STDOUT,
                                preexec_fn=_pdeathsig_preexec)
        try:
            boot = _wait_ready(proc, f"{base_url}/health",
                               float(CONFIG["server_ready_timeout_sec"]), server_log)
            _say(f"server ready in {boot:.0f}s")
            _assert_path_taken(server_log)
            _say(f"path assertion passed: {REQUIRED_BANNER!r} present in the startup log")

            warm_cmd = _bench_argv(python, port, wl, int(CONFIG["warmup_prompts"]), warm_jsonl)
            _say(f"warmup: {CONFIG['warmup_prompts']} prompts (discarded)")
            wrc = subprocess.run(warm_cmd, env=env, check=False).returncode
            if wrc != 0:
                raise SystemExit(f"warmup bench exited rc={wrc}\n--- server log ---\n{_tail(server_log)}")

            bench_cmd = _bench_argv(python, port, wl, int(wl["num_prompts"]), bench_jsonl)
            _say(f"scored: {shlex.join(bench_cmd)}")
            brc = subprocess.run(bench_cmd, env=env, check=False).returncode
            if brc != 0:
                raise SystemExit(f"scored bench exited rc={brc}\n--- server log ---\n{_tail(server_log)}")

            rec = _read_bench_record(bench_jsonl)
            if not rec.get("output_throughput"):
                raise SystemExit(
                    "bench_serving reported no output_throughput. Refusing to fall back to another "
                    "key rather than silently scoring a different quantity.\n"
                    f"keys present: {sorted(rec)}"
                )
            if int(rec.get("completed", 0)) != int(wl["num_prompts"]):
                raise SystemExit(
                    f"bench_serving completed {rec.get('completed')} of {wl['num_prompts']} requests. "
                    "A partial run's throughput is not comparable across arms."
                )

            result = dict(rec)
            result["workload"] = wl
            result["server_boot_sec"] = round(boot, 2)
            out_json.write_text(json.dumps(result, indent=2) + "\n")
            _say(f"output throughput {rec['output_throughput']:.2f} tok/s   "
                 f"mean TPOT {rec.get('mean_tpot_ms', float('nan')):.3f} ms")

            probe = _correctness_probe(base_url)
            try:
                probe_json.write_text(json.dumps(probe, separators=(",", ":")) + "\n")
            except OSError as exc:
                _say(f"correctness probe could not be written: {exc}")
            _say("correctness probe: " + (
                f"{len(probe['prompts'])} prompts in {probe['elapsed_sec']}s, "
                f"greedy digest {probe['output_digest'][:16]}"
                if not probe.get("error") else f"UNAVAILABLE -- {probe['error']}"))
            fp = _numeric_fingerprint(base_url, model)
            (raw / str(NUMERIC_PROBE["file"])).write_text(
                json.dumps({"numeric_fingerprint": fp}, indent=2) + "\n")
            _say("numeric probe: " + (
                f"{fp['n_positions']} positions, sum_logprob {fp['sum_logprob']}"
                if fp.get("available") else f"unavailable -- {fp.get('reason')}"))
            for name in extra_points:
                points_benched.append(_bench_point(
                    python, port, env, name, spec["points"][name],
                    point_raw_dir(raw, name, spec["anchor"]), serve_cmd, server_log))
            rc = 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
            deadline = time.time() + 60
            while _port_busy(port) and time.time() < deadline:
                time.sleep(1.0)
    elapsed = time.time() - t0

    manifest.write_text(json.dumps({
        "mode": "serve",
        "returncode": rc,
        "command": f"sglang bench_serving (vllm bench manifest shape): {shlex.join(bench_cmd or serve_cmd)}",
        "serve_command": shlex.join(serve_cmd),
        "bench_command": shlex.join(bench_cmd) if bench_cmd else "",
        "repo": str(repo),
        "model": model,
        "workload": wl,
        "workloads_benched": [wl["name"]] + [pt["name"] for pt in points_benched if not pt["error"]],
        "points": points_benched,
        "elapsed_sec": round(elapsed, 2),
        "result_json": str(out_json),
    }, indent=2) + "\n", encoding="utf-8")
    _say(f"rc={rc} elapsed={elapsed:.0f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
