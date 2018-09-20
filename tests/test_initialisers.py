"""Test the initialisation functions."""

import numpy as np
import tensorflow as tf

import aboleth as ab


def test_glorot_std():
    result = ab.initialisers._glorot_std(10, 21)
    assert np.allclose(result, 1. / np.sqrt(3 * 31))


def test_autonorm_std():
    result = ab.initialisers._autonorm_std(10, 21)
    assert np.allclose(result, 1. / np.sqrt(10))

def test_initialise_weights(mocker):
    mocker.patch.dict("aboleth.initialisers._INIT_DICT",
                      {"foo": lambda x: "bar"})
    shape = mocker.MagicMock()
    result = ab.initialisers.initialise_weights(shape, "foo")
    assert result == "bar"

    result2 = ab.initialisers.initialise_weights(shape, lambda x: x)
    assert result2 is shape


def test_initialise_stds(mocker):
    mocker.patch.dict("aboleth.initialisers._PRIOR_DICT",
                      {"foo": lambda x, y: y + 10 * x})
    init_val = "foo"
    learn_prior = False
    suffix = "bar"
    std, std0 = ab.initialisers.initialise_stds(1, 2, init_val, learn_prior,
                                                suffix)
    assert std == 12.
    assert std.dtype == np.float32

    init_val = 10.
    learn_prior = True
    std, std0 = ab.initialisers.initialise_stds(1, 2, init_val, learn_prior,
                                                suffix)
    assert std.name == 'prior_std_bar:0'

    tc = tf.test.TestCase()
    with tc.test_session() as sess:
        sess.run(tf.global_variables_initializer())
        assert std.eval() == 10.0
