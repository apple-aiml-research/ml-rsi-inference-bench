#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Compose the 36 fidelity sequences from `corpus.py` and write their token ids.

Authoring step, run once INSIDE the task image (it needs the model's tokenizer and
`openai_harmony`); the reference log-probabilities are filled in afterwards by
`record_reference.py` under torchrun. Never run at grading.

Re-running this **resets every `reference` to null**, by design: the reference values belong to a
specific token sequence, and a rebuild may change the tokens. So a rebuild is always followed by a
full `record_reference.py` pass, which is also why all 36 references come from one recording
session rather than being patched in piecemeal.

    /opt/sglang/bin/python build_sequences.py --out tests/hidden/fidelity

Why the harmony conversations are rendered by `openai_harmony` rather than by hand: the library is
what the engine's own serving path speaks, so the special tokens `<|start|>`, `<|channel|>`,
`<|message|>`, `<|end|>` and `<|return|>` land where the template puts them and not where I think it
does. The plain and long sequences are composed by concatenating corpus passages in a fixed order,
repeating the list as needed, and truncating to an exact token count -- so a sequence is reproducible
from `corpus.py` and this file alone, with no network and no model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import corpus  # noqa: E402


def harmony_encoding():
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding

    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


def render_conversation(enc, effort: str, user: str, analysis_key: str, final_key: str) -> list[int]:
    """One complete assistant turn: system (with a reasoning effort), user, analysis, final.

    `auto_drop_analysis=False` is load-bearing and was found by decoding the output rather than by
    reading the docs. Harmony's default drops an assistant `analysis` message when a `final` follows
    it -- the rule that keeps chain-of-thought out of later turns -- so the first build of these
    sequences carried the `final` channel only, at about half the intended length, and never exercised
    `<|channel|>analysis`. `render_conversation_for_training` drops it too. Measured on
    openai-harmony 0.0.3: default 86 tokens with analysis absent, `auto_drop_analysis=False` 106 tokens
    with `<|start|>assistant<|channel|>analysis<|message|>...<|end|>` present.
    """
    from openai_harmony import (
        Conversation,
        DeveloperContent,
        Message,
        ReasoningEffort,
        RenderConversationConfig,
        Role,
        SystemContent,
    )

    sys_content = SystemContent.new().with_reasoning_effort(
        {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}[effort]
    )
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, sys_content),
            Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions(
                    "Answer from the material given. Think in the analysis channel first."
                ),
            ),
            Message.from_role_and_content(Role.USER, user),
            Message.from_role_and_content(Role.ASSISTANT, corpus.PASSAGES[analysis_key]).with_channel(
                "analysis"
            ),
            Message.from_role_and_content(Role.ASSISTANT, corpus.PASSAGES[final_key]).with_channel(
                "final"
            ),
        ]
    )
    return enc.render_conversation(convo, config=RenderConversationConfig(auto_drop_analysis=False))


def compose(enc, keys: list[str], target: int) -> tuple[list[int], str]:
    """Concatenate the named passages, repeating the list, then truncate to exactly `target` tokens."""
    parts: list[str] = []
    i = 0
    while True:
        text = corpus.SEPARATOR.join(parts)
        if parts and len(enc.encode(text, allowed_special=set())) >= target:
            break
        parts.append(corpus.PASSAGES[keys[i % len(keys)]])
        i += 1
        if i > 400:
            raise RuntimeError(f"cannot reach {target} tokens from {keys}")
    text = corpus.SEPARATOR.join(parts)
    ids = enc.encode(text, allowed_special=set())[:target]
    return ids, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = harmony_encoding()

    index = []
    for name, kind, target, keys in corpus.SEQUENCES:
        ids, text = compose(enc, keys, target)
        assert len(ids) == target, (name, len(ids), target)
        rec = {
            "id": name,
            "kind": kind,
            "construction": {"passages": keys, "separator": "\\n\\n", "target_tokens": target},
            "n_tokens": len(ids),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_ids": ids,
            "reference": None,
        }
        (out / f"{name}.json").write_text(json.dumps(rec) + "\n")
        index.append({"id": name, "kind": kind, "n_tokens": len(ids)})
        print(f"[build] {name:11} {kind:6} {len(ids):5} tokens  {len(keys)} passages")

    for n, (effort, user, analysis, final) in enumerate(corpus.CONVERSATIONS, start=1):
        name = f"harmony_{n:02d}"
        ids = render_conversation(enc, effort, user, analysis, final)
        # Every fourth conversation ends the way the MODEL ends a final message -- with <|return|>
        # rather than the transcript's <|end|>. Harmony writes <|end|> when it renders a completed
        # turn; <|return|> is what the assistant itself emits, and it is a stop token the serving path
        # has to handle. The token id comes from the encoding, so nothing about the format is invented,
        # and the swap is one token at one position.
        returned = n % 4 == 0
        if returned:
            end_id, return_id = enc.encode("<|end|>", allowed_special="all"), enc.encode(
                "<|return|>", allowed_special="all"
            )
            assert len(end_id) == 1 and len(return_id) == 1, (end_id, return_id)
            assert ids[-1] == end_id[0], (name, ids[-3:])
            ids = ids[:-1] + return_id
        rec = {
            "id": name,
            "kind": "hidden",
            "construction": {"harmony": True, "reasoning_effort": effort, "user": user,
                             "analysis_passage": analysis, "final_passage": final,
                             "auto_drop_analysis": False, "ends_with_return_token": returned},
            "n_tokens": len(ids),
            "token_ids": ids,
            "reference": None,
        }
        (out / f"{name}.json").write_text(json.dumps(rec) + "\n")
        index.append({"id": name, "kind": "hidden", "n_tokens": len(ids)})
        print(f"[build] {name:11} hidden {len(ids):5} tokens  effort={effort}"
              f"{'  ends <|return|>' if returned else ''}")

    hidden = [x for x in index if x["kind"] == "hidden"]
    public = [x for x in index if x["kind"] == "public"]
    short = [x for x in index if x["n_tokens"] <= 128]
    assert not short, f"every sequence must exceed the 128-token window: {short}"
    (out / "index.json").write_text(json.dumps(
        {"hidden": len(hidden), "public": len(public), "sequences": index,
         "corpus_sha256": hashlib.sha256(pathlib.Path(corpus.__file__).read_bytes()).hexdigest(),
         "builder_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}, indent=1) + "\n")
    print(f"[build] {len(hidden)} hidden + {len(public)} public sequences; "
          f"shortest {min(x['n_tokens'] for x in index)} tokens")
    return 0 if (len(hidden) == 32 and len(public) == 4) else 1


if __name__ == "__main__":
    sys.exit(main())
