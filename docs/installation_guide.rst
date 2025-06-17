.. _installation_guide:

Installation
============
This section presents how to install the Canari libraty using either PyPi, or through a local installation for developpement purposes. Note that prior to installing, you should first create a Miniconda environement in order to ensure the compatibility with the `pyTAGI <https://github.com/lhnguyen102/cuTAGI>`_ external library.

Create Miniconda Environment
----------------------------

1. Install Miniconda by following these `instructions <https://docs.conda.io/en/latest/miniconda.html>`_.
2. Create a conda environment named ``canari``:

   .. code-block:: sh

      conda create --name canari python=3.10

3. Activate conda environment:

   .. code-block:: sh

      conda activate canari

Installing from PyPi
----------------

.. code-block:: sh

    pip install pycanari

Installing locally
----------------

1. Clone this repository:

   .. code-block:: sh

      git clone https://github.com/Bayes-Works/canari.git
      cd canari

2. Create conda environment following the above instructions

3. Install requirements

   .. code-block:: sh

      pip install -r requirements.txt

4. Install pycanari package

   .. code-block:: sh

      pip install .

