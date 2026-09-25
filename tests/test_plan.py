"""Test the memory planner: sizes, budgets, batches and tiers."""

import numpy as np
import pytest

import src_method._plan as plan_module
from src_method import Resources
from src_method._plan import parse_size, resolve_budgets

GB = 10**9


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (123, 123),
        ("36GB", 36 * 10**9),
        ("36GiB", 36 * 2**30),
        ("1.5 kB", 1500),
        ("512", 512),
        ("2mib", 2 * 2**20),
    ],
)
def test_parse_size(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", [-1, "36 GBs", "lots", ""])
def test_parse_size_rejects_values(value):
    with pytest.raises(ValueError, match="Expected"):
        parse_size(value)


@pytest.mark.parametrize("value", [True, 1.5, None])
def test_parse_size_rejects_types(value):
    with pytest.raises(TypeError, match="Expected"):
        parse_size(value)


def test_resources_validate_at_construction():
    with pytest.raises(ValueError, match="size string"):
        Resources(gpu_memory="plenty")


def test_resolve_budgets_explicit_on_host(tmp_path):
    budgets = resolve_budgets(Resources(host_memory="2GB", scratch_dir=tmp_path), np)

    assert budgets.unified
    assert budgets.host == budgets.device == 2 * GB
    assert budgets.scratch_dir == tmp_path
    assert budgets.disk > 0


def test_resolve_budgets_detects_host_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(plan_module, "host_memory_available", lambda: 10 * GB)

    budgets = resolve_budgets(Resources(scratch_dir=tmp_path / "not" / "yet"), np)

    assert budgets.host == 9 * GB


def test_resolve_budgets_detects_device_memory(monkeypatch, tmp_path):
    fake_xp = object()
    monkeypatch.setattr(plan_module, "is_host", lambda _xp: False)
    monkeypatch.setattr(plan_module, "device_memory", lambda _xp: (30 * GB, 40 * GB))

    budgets = resolve_budgets(
        Resources(host_memory="1GB", scratch_dir=tmp_path), fake_xp
    )

    assert not budgets.unified
    assert budgets.device == 26 * GB  # minus max(10% of 40 GB, 1 GiB)


def test_resolve_budgets_detects_disk(monkeypatch, tmp_path):
    usage = type("Usage", (), {"free": 100 * GB})
    monkeypatch.setattr(plan_module.shutil, "disk_usage", lambda _path: usage)

    budgets = resolve_budgets(Resources(host_memory="1GB", scratch_dir=tmp_path), np)

    assert budgets.disk == 95 * GB
