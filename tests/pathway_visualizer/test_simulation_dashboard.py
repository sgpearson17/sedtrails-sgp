import numpy as np

from sedtrails.pathway_visualizer.simulation_dashboard import SimulationDashboard


class FakeArtist:
    """Minimal matplotlib-like artist used for cleanup and legend tests."""

    def __init__(self, axis=None):
        """Stores an optional axis reference to mimic matplotlib artist state."""
        self.axes = axis

    def remove(self):
        """Marks the artist as detached from its axis."""
        self.axes = None


class FakeImage(FakeArtist):
    """Image-like artist stub supporting imshow update methods."""

    def set_data(self, image):
        """Accepts raster image updates (no-op in tests)."""
        pass

    def set_extent(self, extent):
        """Accepts extent updates (no-op in tests)."""
        pass

    def set_clim(self, vmin=None, vmax=None):
        """Accepts color-limit updates (no-op in tests)."""
        pass


class NonRemovableArtist:
    """Artist stub that fails on remove to test robust cleanup behavior."""

    def remove(self):
        """Raises to emulate third-party artists that cannot be removed."""
        raise NotImplementedError('cannot remove artist')


class FakeAxis:
    """Axis stub recording plotting calls to verify rendering code paths."""

    def __init__(self):
        """Initializes counters and flags used by assertions."""
        self.scatter_sizes = []
        self.scatter_calls = []
        self.plot_lines = []
        self.line_collections = []
        self.quiver_size = None
        self.imshow_shapes = []
        self.tricontourf_called = False
        self.tricontour_called = False

    def clear(self):
        pass

    def scatter(self, x, y, *args, **kwargs):
        x = np.asarray(x)
        y = np.asarray(y)
        self.scatter_sizes.append(len(x))
        self.scatter_calls.append({'x': x.copy(), 'y': y.copy(), 'args': args, 'kwargs': kwargs})
        return FakeArtist(self)

    def plot(self, x, y, *args, **kwargs):
        x = np.asarray(x)
        y = np.asarray(y)
        self.plot_lines.append({'x': x.copy(), 'y': y.copy(), 'args': args, 'kwargs': kwargs})
        return (FakeArtist(self),)

    def add_collection(self, collection):
        self.line_collections.append(
            {
                'segments': np.asarray(collection.get_segments(), dtype=float),
                'alpha': collection.get_alpha(),
                'label': collection.get_label(),
            }
        )
        return collection

    def quiver(self, x, y, u, v, *args, **kwargs):
        self.quiver_size = len(x)
        return FakeArtist(self)

    def imshow(self, image, *args, **kwargs):
        self.imshow_shapes.append(image.shape)
        return FakeImage(self)

    def tricontourf(self, *args, **kwargs):
        self.tricontourf_called = True

    def tricontour(self, *args, **kwargs):
        self.tricontour_called = True

    def set_xlabel(self, label):
        pass

    def set_ylabel(self, label):
        pass

    def set_aspect(self, aspect):
        pass

    def set_title(self, title, *args, **kwargs):
        pass

    def get_legend(self):
        return None

    def legend(self, *args, **kwargs):
        return FakeArtist(self)


def _dashboard_with_axis(axis_name, axis):
    """Creates a minimally configured dashboard with one injected axis."""
    dashboard = object.__new__(SimulationDashboard)
    dashboard.axes = {axis_name: axis}
    dashboard.bathymetry_cmap = 'viridis'
    dashboard.bathymetry_vmin = -12
    dashboard.bathymetry_vmax = 6
    return dashboard


def _flow_field(n_points):
    """Builds a simple synthetic flow-field dictionary with n_points samples."""
    x = np.arange(n_points, dtype=float)
    y = np.zeros(n_points, dtype=float)
    return {
        'x': x,
        'y': y,
        'magnitude': np.ones(n_points),
        'u': np.ones(n_points),
        'v': np.zeros(n_points),
    }


def test_dashboard_should_update_uses_plot_interval():
    """Checks should_update triggers once the elapsed interval is reached."""
    dashboard = object.__new__(SimulationDashboard)
    dashboard.last_update_time = 100.0

    assert not dashboard.should_update(129.0, 30.0)
    assert dashboard.should_update(130.0, 30.0)


def test_spatial_artist_cleanup_ignores_already_cleared_artists():
    """Ensures artist cleanup ignores removal errors and clears the container."""
    dashboard = object.__new__(SimulationDashboard)
    dashboard._particle_artists = [NonRemovableArtist()]

    dashboard._remove_spatial_artist('_particle_artists')

    assert dashboard._particle_artists == []


def test_large_grid_flowfield_plot_uses_raster_path():
    """Verifies large flow fields use raster + decimated quiver rendering."""
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('flowfield', axis)
    n_points = SimulationDashboard.LARGE_GRID_POINT_LIMIT + 1

    dashboard._update_flowfield_plot(_flow_field(n_points), np.zeros(n_points))

    assert axis.imshow_shapes
    assert axis.scatter_sizes == []
    assert axis.quiver_size == SimulationDashboard.LARGE_GRID_QUIVER_LIMIT
    assert not axis.tricontourf_called
    assert not axis.tricontour_called


def test_small_grid_flowfield_plot_keeps_contour_path():
    """Verifies small flow fields keep the contour-based rendering path."""
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('flowfield', axis)
    n_points = SimulationDashboard.LARGE_GRID_POINT_LIMIT

    dashboard._update_flowfield_plot(_flow_field(n_points), np.zeros(n_points))

    assert axis.scatter_sizes == []
    assert axis.tricontourf_called
    assert axis.tricontour_called


def test_large_grid_bathymetry_plot_uses_raster_path():
    """Verifies large bathymetry grids switch from contours to raster rendering."""
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('bathymetry', axis)
    n_points = SimulationDashboard.LARGE_GRID_POINT_LIMIT + 1
    particles = {'x': np.array([]), 'y': np.array([])}

    dashboard._update_bathymetry_plot(_flow_field(n_points), np.zeros(n_points), particles)

    assert axis.imshow_shapes
    assert axis.scatter_sizes == []
    assert not axis.tricontourf_called
    assert not axis.tricontour_called


def test_bathymetry_plot_hides_left_domain_and_marks_stranded_particles():
    """Left-domain particles are hidden; stranded particles use the light-red layer."""
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('bathymetry', axis)
    particles = {
        'x': np.array([0.0, 1.0, 2.0, 3.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.0]),
        'x_initial': np.array([10.0, 11.0, 12.0, 13.0]),
        'y_initial': np.array([1.0, 1.0, 1.0, 1.0]),
        'status_left_domain': np.array([False, True, False, False]),
        'status_beached': np.array([False, False, True, False]),
    }

    dashboard._update_bathymetry_plot(_flow_field(4), np.zeros(4), particles)

    assert axis.scatter_sizes == [2, 1, 3]
    np.testing.assert_array_equal(axis.scatter_calls[0]['x'], np.array([0.0, 3.0]))
    np.testing.assert_array_equal(axis.scatter_calls[1]['x'], np.array([2.0]))
    np.testing.assert_array_equal(axis.scatter_calls[2]['x'], np.array([10.0, 12.0, 13.0]))
    assert axis.scatter_calls[1]['kwargs']['color'] == SimulationDashboard.STRANDED_PARTICLE_COLOR
    assert axis.scatter_calls[1]['kwargs']['label'] == 'Stranded'
    assert axis.plot_lines == []
    assert len(axis.line_collections) == 1
    assert axis.line_collections[0]['segments'].shape == (3, 2, 2)
    assert np.all(axis.line_collections[0]['segments'][:, :, 0] != 1.0)


def test_bathymetry_plot_draws_visible_particle_trajectory_history():
    """Visible particles use stored dashboard snapshots for trajectory trails."""
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('bathymetry', axis)
    dashboard.trajectories = {
        'x': [np.array([0.0, 10.0]), np.array([1.0, 11.0]), np.array([2.0, 12.0])],
        'y': [np.array([0.0, 20.0]), np.array([1.0, 21.0]), np.array([2.0, 22.0])],
        'time': [0.0, 1.0, 2.0],
    }
    particles = {
        'x': np.array([2.0, 12.0]),
        'y': np.array([2.0, 22.0]),
        'x_initial': np.array([0.0, 10.0]),
        'y_initial': np.array([0.0, 20.0]),
        'status_left_domain': np.array([False, True]),
        'status_beached': np.array([False, False]),
    }

    dashboard._update_bathymetry_plot(_flow_field(4), np.zeros(4), particles)

    assert axis.plot_lines == []
    assert len(axis.line_collections) == 1
    segments = axis.line_collections[0]['segments']
    np.testing.assert_array_equal(segments[0, :, 0], np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(segments[0, :, 1], np.array([0.0, 1.0, 2.0]))
    assert 10.0 not in segments[:, :, 0]


def test_rasterization_reuses_cached_weights_for_same_grid():
    """Ensures rasterization weight lookup is cached and reused for same mesh."""
    dashboard = object.__new__(SimulationDashboard)
    x = np.array([0.5])
    y = np.array([0.5])
    mesh_geometry = {
        'node_x': np.array([0.0, 1.0, 1.0, 0.0]),
        'node_y': np.array([0.0, 0.0, 1.0, 1.0]),
        'face_node_connectivity': np.array([[0, 1, 2, 3]], dtype=np.int64),
    }

    weights = dashboard._get_raster_weights(x, y, mesh_geometry)
    image = dashboard._rasterize_field(np.array([7.0]), weights)
    same_weights = dashboard._get_raster_weights(x, y, mesh_geometry)

    assert same_weights is weights
    assert np.nanmin(image) == 7.0
    assert np.nanmax(image) == 7.0
