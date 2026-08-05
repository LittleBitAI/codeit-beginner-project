"""Train capability가 없거나 깨졌을 때의 Web 호환 동작."""

from __future__ import annotations

from src.pipelines.web import train_capabilities


def test_missing_capability_uses_the_current_fixed_train_configuration():
    capability = train_capabilities.resolve_train_capability(None)

    assert capability == {
        "schema_version": 1,
        "source": "legacy_fallback",
        "fallback_reason": "train_capability_unavailable",
        "model": {
            "default": train_capabilities.LEGACY_ARCHITECTURE,
            "choices": [train_capabilities.LEGACY_ARCHITECTURE],
            "selection_supported": False,
        },
        "optimizer": {
            "default": "SGD",
            "choices": ["SGD"],
            "selection_supported": False,
        },
    }


def test_valid_capability_is_normalized_without_mutating_it():
    reported = {
        "schema_version": 1,
        "model": {"default": "detector_a", "choices": ["detector_a", "detector_b"]},
        "optimizer": {"default": "AdamW", "choices": ["AdamW", "SGD", "AdamW"]},
    }

    resolved = train_capabilities.resolve_train_capability(reported)

    assert resolved == {
        "schema_version": 1,
        "source": "train",
        "fallback_reason": None,
        "model": {
            "default": "detector_a",
            "choices": ["detector_a", "detector_b"],
            "selection_supported": True,
        },
        "optimizer": {
            "default": "AdamW",
            "choices": ["AdamW", "SGD"],
            "selection_supported": True,
        },
    }
    assert reported["optimizer"]["choices"] == ["AdamW", "SGD", "AdamW"]


def test_invalid_or_unknown_capability_version_fails_closed_to_fallback():
    for reported in (
        {"schema_version": 2, "model": {}, "optimizer": {}},
        {"schema_version": 1, "model": {"default": "x", "choices": []}},
        {"schema_version": 1, "model": {"default": "../x", "choices": ["../x"]}},
    ):
        resolved = train_capabilities.resolve_train_capability(reported)

        assert resolved["source"] == "legacy_fallback"
        assert resolved["fallback_reason"] == "train_capability_invalid"


def test_defaults_api_reports_fallback_source(client):
    body = client.get("/api/train/defaults").json()

    assert body["architecture"] == train_capabilities.LEGACY_ARCHITECTURE
    assert body["train_capability"]["source"] == "legacy_fallback"
    assert body["train_capability"]["optimizer"]["default"] == "SGD"


def test_defaults_api_accepts_a_future_reported_capability(client, monkeypatch):
    monkeypatch.setattr(
        train_capabilities,
        "reported_train_capability",
        lambda: {
            "schema_version": 1,
            "model": {"default": "detector_a", "choices": ["detector_a"]},
            "optimizer": {"default": "AdamW", "choices": ["AdamW"]},
        },
    )

    body = client.get("/api/train/defaults").json()

    assert body["architecture"] == "detector_a"
    assert body["train_capability"]["source"] == "train"
    assert body["train_capability"]["optimizer"]["default"] == "AdamW"
