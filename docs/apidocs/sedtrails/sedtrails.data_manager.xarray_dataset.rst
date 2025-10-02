:py:mod:`sedtrails.data_manager.xarray_dataset`
===============================================

.. py:module:: sedtrails.data_manager.xarray_dataset

.. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`create_sedtrails_dataset <sedtrails.data_manager.xarray_dataset.create_sedtrails_dataset>`
     - .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.create_sedtrails_dataset
          :summary:
   * - :py:obj:`populate_population_metadata <sedtrails.data_manager.xarray_dataset.populate_population_metadata>`
     - .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.populate_population_metadata
          :summary:
   * - :py:obj:`populate_flowfield_metadata <sedtrails.data_manager.xarray_dataset.populate_flowfield_metadata>`
     - .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.populate_flowfield_metadata
          :summary:
   * - :py:obj:`collect_timestep_data <sedtrails.data_manager.xarray_dataset.collect_timestep_data>`
     - .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.collect_timestep_data
          :summary:

API
~~~

.. py:function:: create_sedtrails_dataset(N_particles, N_populations, N_timesteps, N_flowfields, name_strlen=24)
   :canonical: sedtrails.data_manager.xarray_dataset.create_sedtrails_dataset

   .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.create_sedtrails_dataset

.. py:function:: populate_population_metadata(ds, populations)
   :canonical: sedtrails.data_manager.xarray_dataset.populate_population_metadata

   .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.populate_population_metadata

.. py:function:: populate_flowfield_metadata(ds, flow_field_names)
   :canonical: sedtrails.data_manager.xarray_dataset.populate_flowfield_metadata

   .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.populate_flowfield_metadata

.. py:function:: collect_timestep_data(ds, populations, timestep, current_time)
   :canonical: sedtrails.data_manager.xarray_dataset.collect_timestep_data

   .. autodoc2-docstring:: sedtrails.data_manager.xarray_dataset.collect_timestep_data
