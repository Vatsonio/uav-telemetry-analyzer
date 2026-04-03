"""Тести для модуля coordinates."""
import numpy as np
import pytest
from src.coordinates import haversine, total_distance_haversine, wgs84_to_enu


class TestHaversine:
    def test_same_point(self):
        assert haversine(50.0, 30.0, 50.0, 30.0) == 0.0

    def test_known_distance(self):
        # Київ → Львів ~470 км
        d = haversine(50.45, 30.52, 49.84, 24.03)
        assert 460_000 < d < 480_000

    def test_small_distance(self):
        # ~111 м на 0.001 градуса по широті
        d = haversine(50.0, 30.0, 50.001, 30.0)
        assert 100 < d < 120

    def test_symmetry(self):
        d1 = haversine(50.0, 30.0, 51.0, 31.0)
        d2 = haversine(51.0, 31.0, 50.0, 30.0)
        assert abs(d1 - d2) < 0.01


class TestTotalDistance:
    def test_single_point(self):
        assert total_distance_haversine(np.array([50.0]), np.array([30.0])) == 0.0

    def test_straight_line(self):
        lats = np.array([50.0, 50.001, 50.002])
        lons = np.array([30.0, 30.0, 30.0])
        d = total_distance_haversine(lats, lons)
        assert 200 < d < 240

    def test_roundtrip(self):
        lats = np.array([50.0, 50.001, 50.0])
        lons = np.array([30.0, 30.0, 30.0])
        d = total_distance_haversine(lats, lons)
        single = haversine(50.0, 30.0, 50.001, 30.0)
        assert abs(d - 2 * single) < 0.01


class TestWGS84toENU:
    def test_origin_is_zero(self):
        e, n, u = wgs84_to_enu(
            np.array([50.0]), np.array([30.0]), np.array([100.0]),
            50.0, 30.0, 100.0,
        )
        assert abs(e[0]) < 1e-6
        assert abs(n[0]) < 1e-6
        assert abs(u[0]) < 1e-6

    def test_east_direction(self):
        e, n, u = wgs84_to_enu(
            np.array([50.0]), np.array([30.001]), np.array([100.0]),
            50.0, 30.0, 100.0,
        )
        assert e[0] > 0  # на схід
        assert abs(n[0]) < 1e-6  # широта не змінилась

    def test_north_direction(self):
        e, n, u = wgs84_to_enu(
            np.array([50.001]), np.array([30.0]), np.array([100.0]),
            50.0, 30.0, 100.0,
        )
        assert abs(e[0]) < 1e-6
        assert n[0] > 0  # на північ

    def test_up_direction(self):
        e, n, u = wgs84_to_enu(
            np.array([50.0]), np.array([30.0]), np.array([150.0]),
            50.0, 30.0, 100.0,
        )
        assert abs(u[0] - 50.0) < 1e-6
