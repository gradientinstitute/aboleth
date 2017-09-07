.. _install:

Installation
============

Firstly, make sure you have `TensorFlow <https://www.tensorflow.org/>`_
installed, preferably compiled specifically for your architecture, see
`installing TensorFlow from sources
<https://www.tensorflow.org/install/install_sources>`_.

To get up and running quickly you can use pip and get the Aboleth package from
`PyPI <https://pypi.python.org/pypi>`_::

    $ pip install aboleth

Or, to install additional dependencies required by the `demos
<https://github.com/data61/aboleth/tree/develop/demos>`_::

    $ pip install aboleth[demos]

To install in develop mode with packages required for development we recommend
you clone the repository from GitHub::

    $ git clone git@github.com:data61/aboleth.git

Then in the directory that you cloned into, issue the following::

    $ pip install -e .[dev]

Or::

    $ pip install -e .[dev,demos]

If you also want to develop with the demos.
