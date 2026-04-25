import numpy as np

from sedtrails.pathway_visualizer.simulation_dashboard import SimulationDashboard


class FakeArtist:
    def __init__(self, axis=None):
        self.axes = axis

    def remove(self):
        self.axes = None


class FakeImage(FakeArtist):
    def set_data(self, image):
        pass

    def set_extent(self, extent):
        pass

    def set_clim(self, vmin=None, vmax=None):
        pass


class NonRemovableArtist:
    def remove(self):
        raise NotImplementedError('cannot remove artist')


class FakeAxis:
    def __init__(self):
        self.scatter_sizes = []
        self.quiver_size = None
        self.imshow_shapes = []
        self.tricontourf_called = False
        self.tricontour_called = False

    def clear(self):
        pass

    def scatter(self, x, y, *args, **kwargs):
        self.scatter_sizes.append(len(x))
        return FakeArtist(self)

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
    dashboard = object.__new__(SimulationDashboard)
    dashboard.axes = {axis_name: axis}
    dashboard.bathymetry_cmap = 'viridis'
    dashboard.bathymetry_vmin = -12
    dashboard.bathymetry_vmax = 6
    return dashboard


def _flow_field(n_points):
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
    dashboard = object.__new__(SimulationDashboard)
    dashboard.last_update_time = 100.0

    assert not dashboard.should_update(129.0, 30.0)
    assert dashboard.should_update(130.0, 30.0)


def test_spatial_artist_cleanup_ignores_already_cleared_artists():
    dashboard = object.__new__(SimulationDashboard)
    dashboard._particle_artists = [NonRemovableArtist()]

    dashboard._remove_spatial_artist('_particle_artists')

    assert dashboard._particle_artists == []


def test_large_grid_flowfield_plot_uses_raster_path():
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
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('flowfield', axis)
    n_points = SimulationDashboard.LARGE_GRID_POINT_LIMIT

    dashboard._update_flowfield_plot(_flow_field(n_points), np.zeros(n_points))

    assert axis.scatter_sizes == []
    assert axis.tricontourf_called
    assert axis.tricontour_called


def test_large_grid_bathymetry_plot_uses_raster_path():
    axis = FakeAxis()
    dashboard = _dashboard_with_axis('bathymetry', axis)
    n_points = SimulationDashboard.LARGE_GRID_POINT_LIMIT + 1
    particles = {'x': np.array([]), 'y': np.array([])}

    dashboard._update_bathymetry_plot(_flow_field(n_points), np.zeros(n_points), particles)

    assert axis.imshow_shapes
    assert axis.scatter_sizes == []
    assert not axis.tricontourf_called
    assert not axis.tricontour_called


def test_rasterization_reuses_cached_weights_for_same_grid():
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
