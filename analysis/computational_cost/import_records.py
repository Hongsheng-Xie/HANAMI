"""Copy historical cost records without logs/checkpoints; redact local path roots.

This is an archival operation, not a training run or timing-quality selection.
Original and published SHA256 values record the path-only normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def redact(value, mappings):
    if isinstance(value, str):
        for old, new in mappings:
            value = value.replace(old, new).replace(old.replace("\\", "/"), new)
        return value
    if isinstance(value, list):
        return [redact(item, mappings) for item in value]
    if isinstance(value, dict):
        return {redact(key, mappings): redact(item, mappings) for key, item in value.items()}
    return value


def import_directory(source, destination, mappings):
    if destination.exists():
        raise ValueError(f"Archive destination already exists: {destination}")
    inventory = []
    for source_file in sorted(source.rglob("*")):
        if not source_file.is_file() or source_file.suffix not in (".json", ".jsonl"):
            continue
        raw = source_file.read_bytes()
        text = raw.decode("utf-8-sig")
        if source_file.suffix == ".jsonl":
            # A partial final line is an archival error, never silently dropped.
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
            published = "\n".join(json.dumps(redact(row, mappings)) for row in values) + "\n"
        else:
            published = json.dumps(redact(json.loads(text), mappings), indent=2) + "\n"
        relative = source_file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(published, encoding="utf-8", newline="\n")
        inventory.append(dict(path=relative.as_posix(), original_sha256=digest(raw),
                              published_sha256=digest(target.read_bytes())))
    (destination / "archive_inventory.json").write_text(json.dumps(dict(
        source_directory_name=source.name,
        transformation="JSON reserialization and local path-root substitution only; numeric values unchanged.",
        files=inventory), indent=2) + "\n", encoding="utf-8")
    return len(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--replace-root", nargs=2, action="append", default=[], metavar=("PATH", "TOKEN"))
    args = parser.parse_args()
    mappings = sorted(args.replace_root, key=lambda pair: len(pair[0]), reverse=True)
    print(f"Archived {import_directory(args.source, args.destination, mappings)} record files.")


if __name__ == "__main__":
    main()
