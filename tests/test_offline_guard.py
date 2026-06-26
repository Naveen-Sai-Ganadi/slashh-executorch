"""M10 — airplane-mode hardening, enforced as a test.

The on-device app must do everything (mic -> score -> calming intervention)
without networking: no INTERNET permission, and no network APIs anywhere in the
Android sources. These tests grep the app tree so a regression (someone adds an
HTTP client, an analytics SDK, or the INTERNET permission) fails CI instead of
silently shipping a device that can phone home.

Scope is the on-device app (`android/`). Build-time tooling that legitimately
talks to the network (e.g. Qualcomm AI Hub profiling in `model/`) is out of
scope by design — it never runs on the phone.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANDROID = ROOT / "android"
MANIFEST = ANDROID / "app" / "src" / "main" / "AndroidManifest.xml"

# Permissions that would let the app reach the network at all.
FORBIDDEN_PERMISSIONS = (
    "android.permission.INTERNET",
    "android.permission.ACCESS_NETWORK_STATE",
    "android.permission.ACCESS_WIFI_STATE",
)

# Network/IO surfaces that have no business in an offline audio app. Matched as
# regexes against the Kotlin sources (comments included — even a stray import is
# a signal worth failing on).
FORBIDDEN_PATTERNS = (
    r"\bimport\s+java\.net\.",
    r"\bimport\s+javax\.net\.",
    r"\bimport\s+okhttp3\b",
    r"\bimport\s+retrofit2\b",
    r"\bimport\s+android\.net\.",
    r"\bHttpURLConnection\b",
    r"\bHttpsURLConnection\b",
    r"\.openConnection\s*\(",
    r"\bSocket\s*\(",
    r"\bDatagramSocket\b",
    r"\bWebSocket\b",
)


def _kotlin_sources() -> list[Path]:
    return sorted(ANDROID.rglob("*.kt"))


def test_manifest_exists():
    assert MANIFEST.is_file(), f"missing {MANIFEST}"


def test_no_network_permissions_declared():
    text = MANIFEST.read_text(encoding="utf-8")
    # Only flag an actual <uses-permission ... NAME ... />, not the explanatory
    # comment that documents why INTERNET is intentionally absent.
    declared = re.findall(r"<uses-permission[^>]*android:name=\"([^\"]+)\"", text)
    offenders = [p for p in declared if p in FORBIDDEN_PERMISSIONS]
    assert not offenders, f"network permission(s) declared in manifest: {offenders}"


def test_only_microphone_permission_declared():
    text = MANIFEST.read_text(encoding="utf-8")
    declared = set(re.findall(r"<uses-permission[^>]*android:name=\"([^\"]+)\"", text))
    assert declared == {"android.permission.RECORD_AUDIO"}, (
        f"unexpected permissions declared: {declared - {'android.permission.RECORD_AUDIO'}}"
    )


def test_no_network_apis_in_android_sources():
    compiled = [(p, re.compile(p)) for p in FORBIDDEN_PATTERNS]
    hits: list[str] = []
    for src in _kotlin_sources():
        text = src.read_text(encoding="utf-8")
        for raw, rx in compiled:
            for m in rx.finditer(text):
                line = text[: m.start()].count("\n") + 1
                hits.append(f"{src.relative_to(ROOT)}:{line}: matches /{raw}/")
    assert not hits, "network API surface found in on-device sources:\n" + "\n".join(hits)
