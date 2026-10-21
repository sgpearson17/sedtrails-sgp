Transport Converter
===================

This is an overview of the SedTRAILS transport converter, which is designed to convert sediment transport data into a format suitable for analysis and visualization in the SedTRAILS framework. The transport converter is implemented as a Python package, and the source code is available in the `sedtrails.transport_converter` module. The package includes various plugins for different sediment transport models and physics, allowing users to convert data from these models into the SedTRAILS format.



Format Converter
----------------

.. automodule:: sedtrails.transport_converter.format_converter
                    :members:

Domain Masks and Boundary Classes
---------------------------------

FM, SFINCS, XBeach, and Delft3D-4 NetCDF converters share extent clipping,
``domain.inner_boundary_pol_files`` masking, and
``domain.boundary_class_pol_files`` classification through ``BaseFormatPlugin``
and the model-independent ``domain_mask`` module.
When a ``domain`` block is present in a YAML configuration, it must also define
the active extent with either ``pol_file`` or ``subset_x``/``subset_y``. Omit
the entire ``domain`` block when no custom extent, cutout, or boundary-class
override is needed.

``inner_boundary_pol_files`` points to one or more Tekal ``.pol`` files. Each
file may contain multiple polygon blocks. Candidate mesh faces or triangles with
centroids inside these polygons are excluded from the active particle-location
connectivity, while the original node coordinate arrays are left unchanged. This
turns islands and grid cutouts into holes for particle-domain checks.
``pol_file`` retains centroids inside any polygon block; alternatively,
``subset_x``/``subset_y`` retain centroids within inclusive source-coordinate
ranges. These operations select existing faces, not exact polygon intersections.
Source topology and field indexing are preserved. SFINCS also compacts its
source face fields using the shared point mask before building tracking triangles.

``boundary_class_pol_files`` assigns classes to active boundary edges by testing
edge midpoints against class-specific polygons. Supported classes are
``open`` and ``land``. These polygons can be used with either a ``pol_file``
domain or a ``subset_x``/``subset_y`` domain. They may be wider than a thin
line around the edge because only edge midpoints are tested, but they should
not contain midpoint locations from unrelated neighboring boundary edges. Land
takes priority when an edge is selected by both classes. The resulting edge
ids, midpoints, assigned classes, source matches, and counts are stored in
``boundary_edge_classification`` metadata and are used by the particle tracer
to decide whether a boundary crossing means leaving the domain or temporary
beaching.

Adding a reader
^^^^^^^^^^^^^^^

New format plugins should inherit ``BaseFormatPlugin`` and call
``super().__init__()``. The reader remains responsible for loading coordinates,
inferring its coordinate system, normalizing authoritative source connectivity,
and keeping forcing values aligned with those coordinates. Do not replace
source topology with a global triangulation when source connectivity is available.

Once candidate connectivity indexes the reader's forcing coordinates:

* Call ``self._filter_domain_connectivity(x, y, connectivity)`` and use the
  returned ``connectivity`` for particle tracking. Polygonal source faces can
  subsequently be split with ``triangulate_face_connectivity``.
* Call ``self._boundary_edge_classification(x, y, tracking_connectivity)`` only
  after filtering. Return this table as ``boundary_edge_classification`` in the
  seeding container and store it in converted metadata with
  ``self._add_boundary_edge_metadata``.
* Use the same geometry-building path for ``convert`` and
  ``get_seeding_field_data``. Add cutout diagnostics with
  ``self._add_inner_boundary_metadata`` after building the active geometry.
* Readers that compact source face-center fields can use
  ``self._domain_point_mask(x, y)``; apply its active mask to every corresponding
  field and remap source topology before building tracking connectivity.

The base class owns polygon loading, configuration validation, boundary
classification, diagnostics, and geometry cache comparisons. Reader caches
must include ``self._domain_config_signature()`` along with their source
topology identity, coordinates, and CRS. The signature includes runtime
geometry, so switching between planar and geodetic masking invalidates caches.
Configured classes selecting no final boundary edges produce a warning,
including when land overrides every selected open edge.

.. automodule:: sedtrails.transport_converter.tekal
                    :members:

.. automodule:: sedtrails.transport_converter.domain_mask
                    :members:

Physics Library
---------------

.. automodule:: sedtrails.transport_converter.physics_lib
                    :members:

Physics Converters
-------------------
.. automodule:: sedtrails.transport_converter.physics_converter
                    :members:

Format Plug-ins
---------------

SedTRAILS currently exposes D-Flow FM, XBeach, and SFINCS input formats through
the public configuration schema. The Delft3D-4 TRIM module is retained in the
codebase as a legacy stub and raises ``NotImplementedError`` if used directly.

D-Flow FM (netcdf)
^^^^^^^^^^^^^^^^^^
.. automodule:: sedtrails.transport_converter.plugins.format.fm_netcdf
                    :members:

XBeach (netcdf)
^^^^^^^^^^^^^^^
.. automodule:: sedtrails.transport_converter.plugins.format.xbeach
                    :members:

SFINCS (netcdf)
^^^^^^^^^^^^^^^
.. automodule:: sedtrails.transport_converter.plugins.format.sfincs
                    :members:


Physics Plug-ins
------------------

Van Westen et al. (2025)
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sedtrails.transport_converter.plugins.physics.van_westen
                    :members:

Soulsby et al. (2011)
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sedtrails.transport_converter.plugins.physics.soulsby
                    :members:
