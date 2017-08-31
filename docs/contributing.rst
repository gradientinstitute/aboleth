.. _contrib:

Contributing Guidelines
=======================

Please contribute if you think a feature is missing in Aboleth, if you think
an implementation could be better or if you can solve an existing issue!

We just request you read the following before making any changes to the
codebase.


Pull Requests
-------------

This is the best way to contribute. We usually follow a `git-flow 
<https://datasift.github.io/gitflow/IntroducingGitFlow.html>`_ based
development cycle. The way this works is quite simple:

1. Make an issue on our github with your proposed feature or fix.
2. Fork, or make a branch name with the issue number like `feature/#113`.
3. When finished, submit a pull request to merge into `develop`, and refer to
   which issue is being closed in the pull request comment (i.e. `closes
   #113`).
4. One of use will review the pull request.
5. If accepted, your feature will be merged into develop.
6. Your change will eventually be merged into master and tagged with a release
   number.


Code and Documentation Style
----------------------------

In Aboleth we are *only* targeting python 3 - our code is much more elegant as
a result, and we don't have the resources to also support python 2, sorry.

We adhere to the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_
convention for code, and the `PEP 257
<https://www.python.org/dev/peps/pep-0257/>`_ convention for docstrings, using
the `NumPy/SciPy documentation
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
style. Our continuous integration automatically runs linting checks, so any
pull-request will automatically fail if these conventions are not followed.

The builtin Sphinx extension Napoleon is used to parse NumPy style docstrings.
To build the documentation you can run ``make`` from the ``docs`` directory
with the html option::

    $ make html


Testing
-------

We use `py.test <https://docs.pytest.org/en/latest/>`_ for all of our unit
testing, most of which lives in the ``tests`` directory, with *judicious* use
of doctests -- i.e. only when they are illustrative of a functions usage. 

All of the dependencies for testing can be installed by issuing::

    $ pip install -e .[dev]

You can run the tests by issuing from the top level repository directory::

    $ pytest .

Our continuous integration (CI) will fail if coverage drops below 90%, and we
generally want coverage to remain significantly above this. Furthermore, our CI
will fail if the code doesn't pass PEP 8 and PEP 257 conventions. You can run
the exact CI tests by issuing::

    $ make coverage
    $ make lint

from the top level repository directory.
