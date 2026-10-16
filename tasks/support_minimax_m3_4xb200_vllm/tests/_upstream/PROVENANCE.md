# Adapted upstream material, with provenance

Upstream is `vllm-project/vllm`. The reference is the landing squash `0a1c5034f5e4fe736db672010cda33d9d850f87e`
(#45381) with the post-merge changes the reference admits; the task's base is the prepared tree over
`a3195fab7b1227e75403fef83891e961e69228a6`. Nothing here is copied into the agent image.

## Parser tests → `tests/infb_verify/test_parsers.py`

| upstream file | commit | git blob | sha256 | functions |
|---|---|---|---|---|
| `tests/reasoning/test_minimax_m3_reasoning_parser.py` | `0a1c5034` (unchanged at `4c626633`, #45744) | `e2cd14562c042c755f8b1b2e134da5d0000c146e` | `5b715bdfb0615e4d…` | 15 |
| `tests/reasoning/test_minimax_m3_reasoning_parser.py` | `d8e422ccda9b39ca8a0756b5b195513045abcee8` (#45718) | `561e4020cb1f401d41691b9c3b69e526fd40f4b5` | `711d7106bb6cd7a4…` | 22 (7 added) |
| `tests/tool_parsers/test_minimax_m3_tool_parser.py` | `0a1c5034`, unchanged through `d8e422cc` | `fd1acabde2e53a367fa3c0b6590572a18a7dfb56` | `cad20df621c1c5a4…` | 6 |

The landing reasoning file has 15 functions, and all 15 are adapted. The seven functions #45718 adds join iff the
reference admits #45718 (`config.json` `infb.parsers.admit_split_marker_functions`, fixed when the task was frozen).

### Scored nodes (four; each the conjunction of its functions, every function's result a JUnit property)

| node | functions |
|---|---|
| `test_reasoning_parser` | `test_parser_registration`, the six `test_nonstreaming_*`, `test_token_id_helpers`, `test_token_id_helpers_enabled_mode`; iff admitted, `test_token_id_helpers_with_split_marker_tokens` |
| `test_reasoning_parser_streaming` | the six landing `test_streaming_*`; iff admitted, the six `test_streaming_split_*` of #45718 |
| `test_tool_parser` | `test_minimax_m3_parser_registered`, `test_non_streaming_nested_tool_call`, `test_non_streaming_without_tool_call_keeps_content`, `test_non_streaming_multiple_tool_calls` |
| `test_tool_parser_streaming` | `test_streaming_without_tool_call_emits_text`, `test_streaming_nested_tool_call` |

Function bodies are upstream's, with the `test_` prefix dropped (they are called by the nodes, not collected), inputs,
chunkings and expected values unchanged.

### Adaptations (every one; nothing else differs)

1. **Obtaining the parsers.** Upstream imports the private classes (`vllm.reasoning.minimax_m3_reasoning_parser.
   MiniMaxM3ReasoningParser`, `vllm.tool_parsers.minimax_m3_tool_parser.MinimaxM3ToolParser`). Adapted:
   `ReasoningParserManager.get_reasoning_parser("minimax_m3")` and `ToolParserManager.get_tool_parser("minimax_m3")`,
   the recipe's parser names, constructed as the serving layer constructs them at the base
   (`vllm/parser/abstract_parser.py`, `vllm/entrypoints/openai/chat_completion/serving.py`): the reasoning parser
   `(tokenizer, chat_template_kwargs=…)`, the tool parser `(tokenizer, tools)` with `tools` positional.
2. **The two registration functions** asserted identity with the private class; adapted to "the manager resolves
   `minimax_m3` to a subclass of the base class" (`vllm.reasoning.ReasoningParser`, `vllm.tool_parsers.ToolParser`).
3. **Tokenizer.** Upstream's fakes (`MiniMaxM3Tokenizer`: the two markers atomic, printable characters one token each;
   `FakeTokenizer`: an empty vocabulary) are replaced by `/model`'s tokenizer through `vllm.tokenizers.get_tokenizer`
   (the served tokenizer: `tokenizer_mode` auto → the checkpoint's `PreTrainedTokenizerFast`). In it `<mm:think>`
   (200059) and `</mm:think>` (200060) are added tokens, so marker text encodes atomically as in upstream's fake; the
   tokenizer has no post-processor, so encodes add no BOS. `tokenizer.encode(...)` is called with
   `add_special_tokens=False` throughout (upstream passes it in most places; its fakes ignore the flag either way).
4. **#45718's split-marker fakes**, reproduced over the real vocabulary. `RuntimeSplitMiniMaxM3Tokenizer` (atomic
   encodes for the parser, `encode_runtime` one token per character for the streamed deltas): the parser gets
   `/model`'s tokenizer and the deltas are encoded one character at a time (`encode_per_character`: each character
   encoded alone, so no added-token match; every character used is ASCII and maps to one byte-level token).
   `SplitMiniMaxM3Tokenizer` (vocabulary has the markers, every encode splits into characters): `SplitMarkerTokenizer`,
   a wrapper over `/model`'s tokenizer whose `encode` and `tokenize` split into characters and which delegates
   everything else (vocabulary, decode, conversions) to the real tokenizer.
5. **End of stream.** `test_streaming_nested_tool_call` ends with an empty delta carrying `EOS_ID = 99`, an arbitrary
   id of the empty fake vocabulary; adapted to `/model`'s `eos_token_id` (200020, `[e~[`).
7. **The streaming reasoning-end state.** Upstream's four streaming functions
   `test_streaming_reasoning_tags_are_not_returned`, `test_streaming_drops_leading_end_tag`,
   `test_streaming_enabled_mode_starts_in_reasoning` and `test_streaming_plain_content_ends_reasoning_phase` assert
   the whole per-step sequence of `is_reasoning_end_streaming`, which stays set once reasoning has ended — a
   behaviour of the reference that the interface's own docstring ("ends in the `delta_ids` on a decode step") and
   base implementation do not give. The adaptation keeps their reasoning and content assertions and, where an end
   tag arrives, only that the state is set on that step (a per-step and a sticky implementation both satisfy it); the
   plain-content function, where no end tag arrives, keeps its content assertions alone.
8. **A repeated parameter key.** Upstream's nested tool-call input carries
   `duplicate_demo`, one key repeated at one level (`<tag>a</tag><tag>b</tag>`), expected back as an array. The
   checkpoint's chat template never renders a repeated key (a list is `<item>` elements), and no served tool call
   on record repeats one; the adaptation drops that element from the input and its entry from the expected
   arguments. Nesting, `<item>` arrays, typed leaves and JSON-string streamed arguments stay asserted.
6. **Markers.** Upstream's `pytestmark` (`cpu_test`, `skip_global_cleanup`) belong to vLLM's own conftest and are
   dropped.

No assertion needed a private attribute, so none is dropped. `streamed_args_for_tool` and `prev_tool_call_arr`
(asserted by `test_streaming_nested_tool_call`) are public attributes of the `ToolParser` base class, set in its
constructor and read by the serving layer when it finishes a streamed tool call (`get_remaining_unstreamed_args`).
