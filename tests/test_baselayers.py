"""Test the baselayers module."""
import tensorflow as tf
import aboleth as ab

from .util import StringLayer, StringMultiLayer


def test_stack2():
    """Test base implementation of stack."""
    def f(X):
        return "f({})".format(X), 10.0

    def g(X):
        return "g({})".format(X), 20.0

    h = ab.baselayers._stack2(f, g)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = h(X="x")
        assert phi == "g(f(x))"
        assert loss.eval() == 30.0


def test_stack2_multi():
    """Test base implementation of stack."""
    def f(X, Y):
        return "f({}, {})".format(X, Y), 10.0

    def g(X):
        return "g({})".format(X), 20.0

    h = ab.baselayers._stack2(f, g)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = h(X="x", Y="y")
        assert phi == "g(f(x, y))"
        assert loss.eval() == 30.0


def test_stack_layer():
    """Test implementation of stack."""
    f = StringLayer("f", 10.0)
    g = StringLayer("g", 20.0)
    h = StringLayer("h", 30.0)
    r = ab.stack(f, g, h)

    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = r("x")
        assert phi == "h(g(f(x)))"
        assert loss.eval() == 60.0


def test_stack_multilayer():
    """Test implementation of stack."""
    f = StringMultiLayer("f", ["x", "y"], 10.0)
    g = StringLayer("g", 20.0)
    h = StringLayer("h", 30.0)
    r = ab.stack(f, g, h)
    tc = tf.test.TestCase()
    with tc.test_session():
        phi, loss = r(x="x", y="y")
        assert phi == "h(g(f(x,y)))"
        assert loss.eval() == 60.0
