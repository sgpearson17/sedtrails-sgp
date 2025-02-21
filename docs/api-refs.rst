SedTRAILS API
================

This is an example about how Autodoc can render Python docstrings.
Autodoc requires the `sphinx.ext.autodoc` extension to be enabled in `conf.py`, 
and this file to be written in `reStructuredText` format.

.. admonition:: Source code must be available when rendering the documentation
   :class: important

   This can be achieved by any of the approces explain here: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#ensuring-the-code-can-be-imported


A Function
-----------

.. autofunction:: sedtrails.mock_api.sum

A Class
-----------

.. autoclass:: sedtrails.mock_api.Calculator
   :members:
