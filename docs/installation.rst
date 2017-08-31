.. _install:

Installation
============

Firstly, make sure you have `TensorFlow <https://www.tensorflow.org/>`_
installed, preferably compiled specifically for your architecture.

At the command line via pip in the project directory::

    $ pip install .

To install additional dependencies required by the `demos <https://github.com/determinant-io/aboleth/tree/develop/demos>`_::

    $ pip install .[demos]

To install in develop mode with packages required for development::

    $ pip install -e .[dev]

Or::

    $ pip install -e .[dev,demos]

If you also want to develop with the demos.
