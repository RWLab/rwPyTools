rwpytools
=========

Python client for the `Robot Wealth <https://robotwealth.com>`_ research
data platform — the equivalent of `rwRTools
<https://github.com/RWLab/rwRtools>`_. History comes from ``rw-api``'s bulk
exports (cached on disk); anything newer than an export's watermark comes
from the matching live endpoint, and :meth:`rwpytools.Client.get` joins the
two.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
