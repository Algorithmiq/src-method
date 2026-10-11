"""Test the host-side behaviour of the backend helpers."""

import numpy as np

from src_method.utils import (
    NullEvent,
    NullStream,
    current_stream,
    device_pool_bytes,
    device_pool_limit,
    host_memory_available,
    is_host,
    new_stream,
    pinned_empty,
    to_device_async,
    to_host_async,
)


def test_host_streams_are_null():
    stream = new_stream(np)

    assert is_host(np)
    assert isinstance(stream, NullStream)
    assert isinstance(current_stream(np), NullStream)
    event = stream.record()
    assert isinstance(event, NullEvent)
    stream.wait_event(event)
    event.synchronize()
    stream.synchronize()


def test_pinned_empty_on_host_is_a_byte_buffer():
    buffer = pinned_empty(24, np)

    assert buffer.dtype == np.uint8
    assert buffer.shape == (24,)


def test_to_device_async_on_host_copies():
    host = np.arange(6.0).reshape(2, 3)

    device = to_device_async(host, np, NullStream())
    host[:] = 0

    np.testing.assert_array_equal(device, np.arange(6.0).reshape(2, 3))


def test_to_host_async_on_host_fills_out():
    out = np.empty((2, 3))

    to_host_async(np.arange(6.0).reshape(2, 3).T.T, out, NullStream())

    np.testing.assert_array_equal(out, np.arange(6.0).reshape(2, 3))


def test_device_pool_limit_is_a_no_op_on_host():
    with device_pool_limit(np, 10):
        pass


def test_device_pool_is_empty_on_host():
    assert device_pool_bytes(np) == 0


def test_host_memory_reads_mem_available(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 100 kB\nMemAvailable:    2048 kB\n")

    assert host_memory_available(str(meminfo)) == 2048 * 1024


def test_host_memory_falls_back_without_meminfo(tmp_path):
    assert host_memory_available(str(tmp_path / "missing")) > 0
