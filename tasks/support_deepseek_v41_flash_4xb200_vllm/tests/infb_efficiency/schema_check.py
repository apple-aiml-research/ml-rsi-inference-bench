#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""A small JSON Schema checker for the core record, standard library only.

It implements exactly the keywords `efficiency.schema.json` uses -- `type`, `const`, `enum`,
`required`, `properties`, `additionalProperties`, `items`, `minItems`, `minimum`,
`exclusiveMinimum`, `pattern`, local `$ref`, `allOf` and `if`/`then` -- so the verifier can check the
record it wrote without a third-party package in the image.

    python3 schema_check.py <record.json> [<schema.json>]     # exit 0 valid, 1 invalid
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = HERE / "efficiency.schema.json"

_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool, "null": type(None),
}


def _is_type(value, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def _resolve(root: dict, ref: str) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def errors(instance, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    root = root if root is not None else schema
    out: list[str] = []
    if "$ref" in schema:
        out += errors(instance, _resolve(root, schema["$ref"]), root, path)
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(instance, n) for n in names):
            return out + [f"{path}: expected {names}, got {type(instance).__name__}"]
    if "const" in schema and instance != schema["const"]:
        out.append(f"{path}: expected {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        out.append(f"{path}: {instance!r} not in {schema['enum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                out.append(f"{path}: missing required {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in instance:
                out += errors(instance[key], sub, root, f"{path}.{key}")
        extra = schema.get("additionalProperties", True)
        for key in instance:
            if key in props:
                continue
            if extra is False:
                out.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                out += errors(instance[key], extra, root, f"{path}.{key}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            out.append(f"{path}: {len(instance)} items, fewer than {schema['minItems']}")
        if "items" in schema:
            for i, item in enumerate(instance):
                out += errors(item, schema["items"], root, f"{path}[{i}]")
    if _is_type(instance, "number"):
        if "minimum" in schema and instance < schema["minimum"]:
            out.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            out.append(f"{path}: {instance} <= exclusive minimum {schema['exclusiveMinimum']}")
    if isinstance(instance, str) and "pattern" in schema and not re.search(schema["pattern"], instance):
        out.append(f"{path}: {instance!r} does not match {schema['pattern']!r}")
    for sub in schema.get("allOf", []):
        out += errors(instance, sub, root, path)
    if "if" in schema and not errors(instance, schema["if"], root, path):
        out += errors(instance, schema.get("then", {}), root, path)
    return out


def validate(instance, schema: dict | None = None) -> list[str]:
    schema = schema if schema is not None else json.loads(SCHEMA.read_text(encoding="utf-8"))
    return errors(instance, schema)


def main(argv: list[str]) -> int:
    record = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    schema = json.loads(Path(argv[2]).read_text(encoding="utf-8")) if len(argv) > 2 else None
    problems = validate(record, schema)
    for p in problems:
        print(f"[schema] {p}")
    print(f"[schema] {'valid' if not problems else f'{len(problems)} problem(s)'}: {argv[1]}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
