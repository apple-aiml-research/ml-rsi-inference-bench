# Recorded numerical examples

`examples.json` contains four public sequences with their original model-author
reference log-probabilities. They illustrate comparison across content families
and prompt lengths:

| Input family | Tokens |
|---|---:|
| Prose | 512 |
| Math | 2,048 |
| Code | 8,192 |
| Structured text | 20,480 |

The metadata records the reference implementation and each example's source.
Compare them with a running server:

```sh
python /app/eval/deepseek_v41_flash/fidelity/check.py --base-url http://127.0.0.1:8000
```

The request retrieves prompt-token log-probabilities through `/v1/completions`.
The first token has no preceding context. The command reports numerical
differences on the remaining tokens without a task pass threshold. These
representative examples make the numerical protocol concrete across short and
long inputs. Evaluation uses held-out inputs within the requested deployment;
passing the examples does not establish complete native support.
