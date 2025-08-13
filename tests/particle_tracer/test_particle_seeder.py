"""
Unit tests for particle seeding strategies.
"""

import pytest
from sedtrails.particle_tracer.particle_seeder import (
    SeedingConfig,
    PointStrategy,
    RandomStrategy,
    GridStrategy,
    TransectStrategy,
)
from sedtrails.exceptions import MissingConfigurationParameter


class TestPointStrategy:
    """Test cases for PointStrategy."""

    def test_point_strategy(self):
        """Test basic point strategy functionality."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                        'quantity': 10,
                    }
                }
            }
        )
        strategy = PointStrategy()
        result = strategy.seed(config)

        assert len(result) == 2
        assert result[0] == (10, 1.0, 2.0)
        assert result[1] == (10, 3.0, 4.0)

    def test_point_strategy_missing_locations(self):
        """Test point strategy with missing locations."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'point': {}},
                        'quantity': 10,
                    }
                }
            }
        )
        strategy = PointStrategy()

        with pytest.raises(MissingConfigurationParameter, match='"locations" must be provided'):
            strategy.seed(config)

    def test_point_strategy_invalid_location_format(self):
        """Test point strategy with invalid location format."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'point': {'locations': ['invalid_format']}},
                        'quantity': 10,
                    }
                }
            }
        )
        strategy = PointStrategy()

        with pytest.raises(ValueError, match='Invalid location string'):
            strategy.seed(config)


class TestRandomStrategy:
    """Test cases for RandomStrategy."""

    def test_random_strategy(self):
        """Test basic random strategy functionality."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0'}},
                        'quantity': 5,
                    }
                }
            }
        )
        strategy = RandomStrategy()
        result = strategy.seed(config)

        assert len(result) == 5
        # Check all particles have quantity 5 and coordinates within bounds
        for qty, x, y in result:
            assert qty == 5
            assert 1.0 <= x <= 3.0
            assert 2.0 <= y <= 4.0

    def test_random_strategy_missing_bbox(self):
        """Test random strategy with missing bounding box."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'random': {}},
                        'quantity': 5,
                    }
                }
            }
        )
        strategy = RandomStrategy()

        with pytest.raises(MissingConfigurationParameter, match='"bbox" must be provided'):
            strategy.seed(config)


class TestGridStrategy:
    """Test cases for GridStrategy."""

    def test_grid_strategy(self):
        """Test basic grid strategy functionality."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {
                            'grid': {
                                'separation': {'dx': 1.0, 'dy': 1.0},
                            }
                        },
                        'quantity': 2,
                    }
                }
            }
        )
        strategy = GridStrategy()
        bbox = {'xmin': 0.0, 'xmax': 2.0, 'ymin': 0.0, 'ymax': 2.0}
        result = strategy.seed(config, bbox=bbox)

        # Should generate a 3x3 grid (0,1,2 in both directions)
        assert len(result) == 9
        # Check first and last points
        assert (2, 0.0, 0.0) in result
        assert (2, 2.0, 2.0) in result

    def test_grid_strategy_no_bbox(self):
        """Test grid strategy without bounding box."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {
                            'grid': {
                                'separation': {'dx': 1.0, 'dy': 1.0},
                            }
                        },
                        'quantity': 2,
                    }
                }
            }
        )
        strategy = GridStrategy()

        with pytest.raises(RuntimeError, match='Bounding box must be provided'):
            strategy.seed(config)

    def test_grid_strategy_missing_separation(self):
        """Test grid strategy with missing separation parameters."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'grid': {}},
                        'quantity': 2,
                    }
                }
            }
        )
        strategy = GridStrategy()
        bbox = {'xmin': 0.0, 'xmax': 2.0, 'ymin': 0.0, 'ymax': 2.0}

        with pytest.raises(MissingConfigurationParameter, match='"grid" must be provided'):
            strategy.seed(config, bbox=bbox)


class TestTransectStrategy:
    """Test cases for TransectStrategy."""

    def test_transect_strategy(self):
        """Test basic transect strategy functionality."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {
                            'transect': {
                                'segments': ['0,0 2,0'],
                                'k': 3,
                            }
                        },
                        'quantity': 5,
                    }
                }
            }
        )
        strategy = TransectStrategy()
        result = strategy.seed(config)

        # Should generate 3 points along the line from (0,0) to (2,0)
        assert len(result) == 3
        assert result[0] == (5, 0.0, 0.0)  # Start point
        assert result[1] == (5, 1.0, 0.0)  # Middle point
        assert result[2] == (5, 2.0, 0.0)  # End point

    def test_transect_strategy_multiple_segments(self):
        """Test transect strategy with multiple segments."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {
                            'transect': {
                                'segments': ['0,0 1,0', '1,0 1,1'],
                                'k': 2,
                            }
                        },
                        'quantity': 1,
                    }
                }
            }
        )
        strategy = TransectStrategy()
        result = strategy.seed(config)

        # Should generate 2 points per segment = 4 total points
        assert len(result) == 4
        # First segment: (0,0) to (1,0)
        assert (1, 0.0, 0.0) in result
        assert (1, 1.0, 0.0) in result
        # Second segment: (1,0) to (1,1)
        assert (1, 1.0, 0.0) in result
        assert (1, 1.0, 1.0) in result

    def test_transect_strategy_missing_segments(self):
        """Test transect strategy with missing segments."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {'transect': {'k': 3}},
                        'quantity': 5,
                    }
                }
            }
        )
        strategy = TransectStrategy()

        with pytest.raises(MissingConfigurationParameter, match='"segments" must be provided'):
            strategy.seed(config)

    def test_transect_strategy_invalid_segment_format(self):
        """Test transect strategy with invalid segment format."""
        config = SeedingConfig(
            {
                'population': {
                    'seeding': {
                        'strategy': {
                            'transect': {
                                'segments': ['invalid_format'],
                                'k': 2,
                            }
                        },
                        'quantity': 1,
                    }
                }
            }
        )
        strategy = TransectStrategy()

        with pytest.raises(ValueError, match='Invalid segment string'):
            strategy.seed(config)
