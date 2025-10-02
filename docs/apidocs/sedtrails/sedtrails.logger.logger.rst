:py:mod:`sedtrails.logger.logger`
=================================

.. py:module:: sedtrails.logger.logger

.. autodoc2-docstring:: sedtrails.logger.logger
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`setup_logging <sedtrails.logger.logger.setup_logging>`
     - .. autodoc2-docstring:: sedtrails.logger.logger.setup_logging
          :summary:
   * - :py:obj:`install_global_excepthook <sedtrails.logger.logger.install_global_excepthook>`
     - .. autodoc2-docstring:: sedtrails.logger.logger.install_global_excepthook
          :summary:
   * - :py:obj:`get_logger <sedtrails.logger.logger.get_logger>`
     - .. autodoc2-docstring:: sedtrails.logger.logger.get_logger
          :summary:
   * - :py:obj:`log_simulation_state <sedtrails.logger.logger.log_simulation_state>`
     - .. autodoc2-docstring:: sedtrails.logger.logger.log_simulation_state
          :summary:
   * - :py:obj:`log_exception <sedtrails.logger.logger.log_exception>`
     - .. autodoc2-docstring:: sedtrails.logger.logger.log_exception
          :summary:
   * - :py:obj:`_format_seeding_strategy <sedtrails.logger.logger._format_seeding_strategy>`
     - .. autodoc2-docstring:: sedtrails.logger.logger._format_seeding_strategy
          :summary:

API
~~~

.. py:function:: setup_logging(output_dir: str, level: str = 'INFO') -> logging.Logger
   :canonical: sedtrails.logger.logger.setup_logging

   .. autodoc2-docstring:: sedtrails.logger.logger.setup_logging

.. py:function:: install_global_excepthook(logger: logging.Logger) -> None
   :canonical: sedtrails.logger.logger.install_global_excepthook

   .. autodoc2-docstring:: sedtrails.logger.logger.install_global_excepthook

.. py:function:: get_logger(name: str | None = None) -> logging.Logger
   :canonical: sedtrails.logger.logger.get_logger

   .. autodoc2-docstring:: sedtrails.logger.logger.get_logger

.. py:function:: log_simulation_state(logger: logging.Logger, state: dict, level=logging.INFO) -> None
   :canonical: sedtrails.logger.logger.log_simulation_state

   .. autodoc2-docstring:: sedtrails.logger.logger.log_simulation_state

.. py:function:: log_exception(logger: logging.Logger, e: Exception, context: str = None) -> None
   :canonical: sedtrails.logger.logger.log_exception

   .. autodoc2-docstring:: sedtrails.logger.logger.log_exception

.. py:function:: _format_seeding_strategy(strategy)
   :canonical: sedtrails.logger.logger._format_seeding_strategy

   .. autodoc2-docstring:: sedtrails.logger.logger._format_seeding_strategy
