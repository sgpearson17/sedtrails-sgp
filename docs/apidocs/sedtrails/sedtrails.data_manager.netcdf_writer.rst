:py:mod:`sedtrails.data_manager.netcdf_writer`
==============================================

.. py:module:: sedtrails.data_manager.netcdf_writer

.. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NetCDFWriter <sedtrails.data_manager.netcdf_writer.NetCDFWriter>`
     - .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter
          :summary:

API
~~~

.. py:class:: NetCDFWriter(output_dir)
   :canonical: sedtrails.data_manager.netcdf_writer.NetCDFWriter

   .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter

   .. rubric:: Initialization

   .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter.__init__

   .. py:method:: _validate_filename(filename)
      :canonical: sedtrails.data_manager.netcdf_writer.NetCDFWriter._validate_filename

      .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter._validate_filename

   .. py:method:: write(xr_dataset, filename, trim_to_actual_timesteps=False, actual_timesteps=None)
      :canonical: sedtrails.data_manager.netcdf_writer.NetCDFWriter.write

      .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter.write

   .. py:method:: create_dataset(N_particles, N_populations, N_timesteps, N_flowfields, name_strlen=24)
      :canonical: sedtrails.data_manager.netcdf_writer.NetCDFWriter.create_dataset

      .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter.create_dataset

   .. py:method:: add_metadata(dataset, populations, flow_field_names, simulation_metadata=None)
      :canonical: sedtrails.data_manager.netcdf_writer.NetCDFWriter.add_metadata

      .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter.add_metadata

   .. py:method:: create_and_write_simulation_results(populations, flow_field_names, N_timesteps, filename='simulation_results.nc', simulation_metadata=None, name_strlen=24)
      :canonical: sedtrails.data_manager.netcdf_writer.NetCDFWriter.create_and_write_simulation_results

      .. autodoc2-docstring:: sedtrails.data_manager.netcdf_writer.NetCDFWriter.create_and_write_simulation_results
