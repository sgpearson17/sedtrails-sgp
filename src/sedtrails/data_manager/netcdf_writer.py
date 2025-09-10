"""
NetCDF Writer
=============
Writes NetCDF files produced by the SedTrails Particle Tracer System.
"""

import xugrid as xu
from pathlib import Path
from datetime import datetime


class NetCDFWriter:
    """
    A class for writing NetCDF files for the SedTrails Particle Tracer System.

    Attributes
    ----------
    output_dir : pathlib.Path
        The directory where output files (NetCDF, images, etc.) are stored.
    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        # If the output directory already exists, we add a timestamp to avoid overwriting
        if output_dir.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = output_dir.parent / f'{output_dir.name}_{timestamp}'
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_filename(self, filename):
        """
        Validates the file name to ensure it is a NetCDF file.
        """
        if not filename.endswith('.nc'):
            raise ValueError('Output file must have a .nc extension.')

    def write(self, ugrid_dataset, filename):
        """
        Write a xu.UgridDataset to a NetCDF file in the output directory.
        We use the "ugrid.to_netcdf()" method to ensure UGRID compliance.

        Parameters
        ----------
        ugrid_dataset : xu.UgridDataset
            The UGRID-compliant dataset to write.
        filename : str
            The name of the NetCDF file to write (should end with .nc).
        """
        self._validate_filename(filename)
        output_path = self.output_dir / filename
        if not isinstance(ugrid_dataset, xu.UgridDataset):
            raise TypeError('Input must be a xu.UgridDataset.')
        ugrid_dataset.ugrid.to_netcdf(output_path)


if __name__ == '__main__':
    # Example usage - Create a dummy UGRID dataset for demonstration
    import numpy as np
    import xarray as xr

    # Create a simple triangular mesh
    # Define nodes (vertices)
    node_x = np.array([0.0, 1.0, 2.0, 0.5, 1.5])
    node_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

    # Define faces (triangles) - 0-based indexing
    face_nodes = np.array(
        [
            [0, 1, 3],  # Triangle 1
            [1, 2, 4],  # Triangle 2
            [1, 3, 4],  # Triangle 3
        ]
    )

    # Create a simple dataset with particle data
    n_particles = 10
    particle_x = np.random.uniform(0, 2, n_particles)
    particle_y = np.random.uniform(0, 1, n_particles)
    particle_depth = np.random.uniform(0, 5, n_particles)

    # Create the dataset
    ds = xr.Dataset(
        {
            'mesh2d_node_x': (['mesh2d_nNodes'], node_x),
            'mesh2d_node_y': (['mesh2d_nNodes'], node_y),
            'mesh2d_face_nodes': (['mesh2d_nFaces', 'mesh2d_nMax_face_nodes'], face_nodes),
            'particle_x': (['nParticles'], particle_x),
            'particle_y': (['nParticles'], particle_y),
            'particle_depth': (['nParticles'], particle_depth),
        },
        coords={
            'mesh2d_nNodes': range(len(node_x)),
            'mesh2d_nFaces': range(len(face_nodes)),
            'mesh2d_nMax_face_nodes': range(face_nodes.shape[1]),
            'nParticles': range(n_particles),
        },
    )

    # Add mesh topology variable
    ds['mesh2d'] = xr.DataArray(
        data=0,
        attrs={
            'cf_role': 'mesh_topology',
            'topology_dimension': 2,
            'node_coordinates': 'mesh2d_node_x mesh2d_node_y',
            'face_node_connectivity': 'mesh2d_face_nodes',
        },
    )

    # Convert to UgridDataset
    ugrid_dataset = xu.UgridDataset(ds)

    # Write the dataset
    writer = NetCDFWriter(output_dir='output')
    writer.write(ugrid_dataset, 'particles_output.nc')
    print(f'Dummy UGRID dataset written to: {writer.output_dir}/particles_output.nc')
