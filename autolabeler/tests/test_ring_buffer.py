"""Tests for the ring buffer module."""

import threading
import time
import numpy as np
import pytest

from autolabeler.acquisition.ring_buffer import RingBuffer


class TestRingBufferBasic:
    """Basic read/write tests."""

    def test_empty_buffer(self):
        buf = RingBuffer(max_samples=100, channels=1)
        assert buf.available_samples == 0
        data, ts = buf.read_latest(10)
        assert data.shape == (0, 1)
        assert ts.shape == (0,)

    def test_single_write_read(self):
        buf = RingBuffer(max_samples=100, channels=1)
        samples = np.array([1.0, 2.0, 3.0])
        timestamps = np.array([0.1, 0.2, 0.3])
        buf.write(samples, timestamps)

        assert buf.available_samples == 3
        assert buf.total_written == 3

        data, ts = buf.read_latest(3)
        np.testing.assert_array_equal(data.flatten(), samples)
        np.testing.assert_array_equal(ts, timestamps)

    def test_multi_channel(self):
        buf = RingBuffer(max_samples=100, channels=3)
        samples = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        timestamps = np.array([0.1, 0.2])
        buf.write(samples, timestamps)

        data, ts = buf.read_latest(2)
        assert data.shape == (2, 3)
        np.testing.assert_array_equal(data, samples)

    def test_read_latest_partial(self):
        buf = RingBuffer(max_samples=100, channels=1)
        samples = np.arange(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.float64) * 0.1
        buf.write(samples, timestamps)

        data, ts = buf.read_latest(5)
        assert data.shape == (5, 1)
        np.testing.assert_array_equal(data.flatten(), [5, 6, 7, 8, 9])

    def test_read_latest_more_than_available(self):
        buf = RingBuffer(max_samples=100, channels=1)
        samples = np.arange(5, dtype=np.float64)
        buf.write(samples)

        data, ts = buf.read_latest(50)
        assert data.shape == (5, 1)


class TestRingBufferWrapAround:
    """Tests for buffer overflow / wrap-around behavior."""

    def test_overflow_keeps_latest(self):
        buf = RingBuffer(max_samples=5, channels=1)
        # Write 8 samples into a buffer of size 5
        samples = np.arange(8, dtype=np.float64)
        timestamps = np.arange(8, dtype=np.float64)
        buf.write(samples, timestamps)

        assert buf.available_samples == 5
        data, ts = buf.read_latest(5)
        np.testing.assert_array_equal(data.flatten(), [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(ts, [3, 4, 5, 6, 7])

    def test_multiple_writes_wrap(self):
        buf = RingBuffer(max_samples=5, channels=1)
        # Write in two batches
        buf.write(np.array([1, 2, 3], dtype=np.float64), np.array([1, 2, 3], dtype=np.float64))
        buf.write(np.array([4, 5, 6], dtype=np.float64), np.array([4, 5, 6], dtype=np.float64))

        data, ts = buf.read_latest(5)
        np.testing.assert_array_equal(data.flatten(), [2, 3, 4, 5, 6])

    def test_massive_overflow(self):
        buf = RingBuffer(max_samples=10, channels=1)
        # Write way more than capacity
        samples = np.arange(100, dtype=np.float64)
        timestamps = np.arange(100, dtype=np.float64)
        buf.write(samples, timestamps)

        data, ts = buf.read_latest(10)
        np.testing.assert_array_equal(data.flatten(), np.arange(90, 100))


class TestRingBufferTimeRange:
    """Tests for time-range queries."""

    def test_time_range_basic(self):
        buf = RingBuffer(max_samples=100, channels=1)
        samples = np.arange(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.float64)
        buf.write(samples, timestamps)

        data, ts = buf.read_time_range(3.0, 7.0)
        np.testing.assert_array_equal(data.flatten(), [3, 4, 5, 6])
        np.testing.assert_array_equal(ts, [3, 4, 5, 6])

    def test_time_range_empty(self):
        buf = RingBuffer(max_samples=100, channels=1)
        data, ts = buf.read_time_range(0.0, 10.0)
        assert data.shape[0] == 0

    def test_time_range_no_match(self):
        buf = RingBuffer(max_samples=100, channels=1)
        samples = np.arange(5, dtype=np.float64)
        timestamps = np.arange(5, dtype=np.float64)
        buf.write(samples, timestamps)

        data, ts = buf.read_time_range(10.0, 20.0)
        assert data.shape[0] == 0


class TestRingBufferThreadSafety:
    """Tests for concurrent read/write."""

    def test_concurrent_write_read(self):
        buf = RingBuffer(max_samples=10000, channels=1)
        errors = []

        def writer():
            for i in range(100):
                samples = np.random.randn(100)
                buf.write(samples)
                time.sleep(0.001)

        def reader():
            for i in range(200):
                try:
                    data, ts = buf.read_latest(50)
                    assert data.shape[1] == 1
                except Exception as e:
                    errors.append(e)
                time.sleep(0.0005)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestRingBufferFromDuration:
    """Tests for factory method."""

    def test_from_duration(self):
        buf = RingBuffer.from_duration(10.0, 1000, channels=2)
        assert buf.max_samples == 10000
        assert buf.channels == 2

    def test_clear(self):
        buf = RingBuffer(max_samples=100, channels=1)
        buf.write(np.arange(50, dtype=np.float64))
        assert buf.available_samples == 50
        buf.clear()
        assert buf.available_samples == 0


class TestRingBufferReadAll:
    """Tests for read_all method."""

    def test_read_all_no_wrap(self):
        buf = RingBuffer(max_samples=100, channels=1)
        samples = np.arange(10, dtype=np.float64)
        timestamps = np.arange(10, dtype=np.float64)
        buf.write(samples, timestamps)

        data, ts = buf.read_all()
        np.testing.assert_array_equal(data.flatten(), samples)

    def test_read_all_with_wrap(self):
        buf = RingBuffer(max_samples=5, channels=1)
        samples = np.arange(8, dtype=np.float64)
        timestamps = np.arange(8, dtype=np.float64)
        buf.write(samples, timestamps)

        data, ts = buf.read_all()
        assert data.shape[0] == 5
        np.testing.assert_array_equal(data.flatten(), [3, 4, 5, 6, 7])
