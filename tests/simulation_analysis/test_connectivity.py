import numpy as np
import matplotlib

matplotlib.use('Agg')

from sedtrails.pathway_visualizer.sedtrails_plotting import TrajectoryArrays
from sedtrails.simulation_analysis.connectivity import (
    adjacency_to_digraph,
    compile_adjacency_matrix,
    compute_network_metrics,
    compute_node_metrics,
    generate_connectivity_polygons,
    plot_adjacency_matrix,
    plot_network_geographic,
    plot_network_layout,
    read_adjacency_netcdf,
    write_adjacency_netcdf,
)


def _sample_trajectories():
    return TrajectoryArrays(
        time=np.tile(np.array([0.0, 1.0, 2.0]), (3, 1)),
        x=np.array(
            [
                [0.0, 1.0, 8.0],
                [0.0, 6.0, 8.0],
                [10.0, 8.0, 2.0],
            ]
        ),
        y=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_generate_connectivity_polygons_per_source_assigns_each_source():
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])

    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')

    assert len(connectivity) == 2
    np.testing.assert_array_equal(connectivity.source_node, np.array([0, 1]))
    assert np.isfinite(connectivity.node_x).all()
    assert np.isfinite(connectivity.node_y).all()


def test_compile_adjacency_all_counts_repeated_particle_positions_by_default():
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    source_node = np.array([0, 0, 1])

    ds = compile_adjacency_matrix(tr, connectivity, source_node=source_node)

    expected = np.array(
        [
            [3.0, 3.0],
            [1.0, 2.0],
        ]
    )
    np.testing.assert_allclose(ds['adjacency'].values, expected)
    assert ds.attrs['compile_mode'] == 'all'
    assert ds.attrs['compile_count_repeated_visits'] == 1


def test_compile_adjacency_all_can_count_unique_particle_sink_visits():
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    source_node = np.array([0, 0, 1])

    ds = compile_adjacency_matrix(
        tr,
        connectivity,
        source_node=source_node,
        count_repeated_visits=False,
    )

    expected = np.array(
        [
            [2.0, 2.0],
            [1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(ds['adjacency'].values, expected)


def test_compile_adjacency_final_and_zero_self_links():
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    source_node = np.array([0, 0, 1])

    ds = compile_adjacency_matrix(
        tr,
        connectivity,
        mode='final',
        source_node=source_node,
        include_self_links=False,
    )

    expected = np.array(
        [
            [0.0, 2.0],
            [1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(ds['adjacency'].values, expected)


def test_compile_adjacency_time_varying_shape_and_probability_rows():
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    source_node = np.array([0, 0, 1])

    ds = compile_adjacency_matrix(
        tr,
        connectivity,
        mode='time',
        source_node=source_node,
        weight='probability',
    )

    assert ds['adjacency'].shape == (3, 2, 2)
    np.testing.assert_allclose(ds['adjacency'].values[0], np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_adjacency_netcdf_roundtrip(tmp_path):
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    ds = compile_adjacency_matrix(tr, connectivity, source_node=np.array([0, 0, 1]))

    path = write_adjacency_netcdf(ds, tmp_path / 'adjacency.nc')
    loaded = read_adjacency_netcdf(path)
    try:
        np.testing.assert_allclose(loaded['adjacency'].values, ds['adjacency'].values)
        assert loaded.attrs['sedtrails_file_kind'] == 'connectivity_adjacency'
    finally:
        loaded.close()


def test_adjacency_to_digraph_and_metrics():
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    ds = compile_adjacency_matrix(tr, connectivity, source_node=np.array([0, 0, 1]))

    graph = adjacency_to_digraph(ds)
    network_metrics = compute_network_metrics(graph)
    node_metrics = compute_node_metrics(graph)

    assert graph.number_of_nodes() == 2
    assert graph[0][1]['weight'] == 3.0
    assert network_metrics['n_edges'] == 4
    assert node_metrics['weighted_out_degree'][0] == 6.0


def test_connectivity_plotting_smoke():
    tr = _sample_trajectories()
    source_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    connectivity = generate_connectivity_polygons(source_xy, mode='per_source')
    ds = compile_adjacency_matrix(tr, connectivity, source_node=np.array([0, 0, 1]))
    graph = adjacency_to_digraph(ds)

    ax_matrix = plot_adjacency_matrix(ds)
    ax_geo = plot_network_geographic(graph)
    ax_layout = plot_network_layout(graph)

    assert ax_matrix.get_xlabel() == 'Sink node'
    assert ax_geo.get_title() == 'Connectivity network'
    assert 'spring' in ax_layout.get_title()
