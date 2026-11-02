"""
Anonymized test bundle: a zip with what someone else needs to understand this
machine's results, and nothing that identifies the person or the card.

Contents: system.vramcard.json, validation.json (the card checked by
`validate`), profiles/*.json from `vramsuite profile`, and manifest.json
(what is inside and what was removed).
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vramsuite import __version__
from vramsuite.core.validate import validate_vramcard

BUNDLE_SCHEMA = "vramsuite_bundle@0.5"
REDACTED = "<redacted>"
# Identify one physical card or one installation, not a kind of setup.
REDACTED_KEYS = {"uuid", "serial", "python_executable"}
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _user_patterns() -> list[tuple[re.Pattern[str], str]]:
    home = os.path.expanduser("~")
    user = os.environ.get("USERNAME") or os.environ.get("USER")
    patterns = []
    if home and home not in ("~", "/"):
        # Both slash styles: JSON written on Windows may use either.
        for variant in {home, home.replace("\\", "/")}:
            patterns.append((re.compile(re.escape(variant), re.IGNORECASE), "~"))
    if user and len(user) >= 3:
        patterns.append((re.compile(re.escape(user), re.IGNORECASE), "<user>"))
    return patterns


def anonymize(data: Any, counter: dict[str, int] | None = None) -> Any:
    """Copy of data with identifying keys removed and home / user name / IPs masked in strings."""
    counter = counter if counter is not None else {}
    patterns = _user_patterns()

    def clean(value: Any, key: str | None = None) -> Any:
        if key in REDACTED_KEYS and value is not None:
            counter[key] = counter.get(key, 0) + 1
            return REDACTED
        if isinstance(value, dict):
            return {item_key: clean(item, item_key) for item_key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            original = value
            for pattern, replacement in patterns:
                value = pattern.sub(replacement, value)
            value = IPV4.sub("<ip>", value)
            if value != original:
                counter["masked_strings"] = counter.get("masked_strings", 0) + 1
        return value

    return clean(data)


def create_bundle(
        output: str | Path,
        card: dict[str, Any],
        profiles_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write the zip and return its manifest."""
    counter: dict[str, int] = {}
    files: dict[str, Any] = {
        "system.vramcard.json": anonymize(card, counter),
        "validation.json": anonymize(validate_vramcard(card), counter),
    }

    profiles = sorted(Path(profiles_dir).glob("*.json")) if profiles_dir and Path(profiles_dir).is_dir() else []
    for path in profiles:
        files[f"profiles/{path.name}"] = anonymize(json.loads(path.read_text(encoding="utf-8")), counter)

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "vramsuite_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files),
        "card_valid": files["validation.json"]["valid"],
        "redacted": {
            "keys": sorted(REDACTED_KEYS),
            "masked": "home folder -> ~, user name -> <user>, IPv4 addresses -> <ip>",
            "counts": counter,
        },
    }
    files["manifest.json"] = manifest

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, json.dumps(content, indent=2, ensure_ascii=False))
    return {**manifest, "path": str(output)}
