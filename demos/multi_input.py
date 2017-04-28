"""Example of using different nets for different input types."""
import tempfile
import urllib.request as req

import numpy as np
import tensorflow as tf
import pandas as pd
import aboleth as ab
from sklearn.metrics import log_loss, accuracy_score


# Data properties
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status",
                       "occupation", "relationship", "race", "gender",
                       "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]
LABEL_COLUMN = "label"


# Algorithm properties
RSEED = 17
CAT_LAYERS = [
    ab.dense_var(output_dim=50, full=False, seed=RSEED)
]
CON_LAYERS = [
    ab.dense_var(output_dim=5, full=True, seed=RSEED)
]
LAYERS = [
    ab.randomArcCosine(100, 1., RSEED),
    ab.dense_var(output_dim=1, full=True, seed=RSEED),
    ab.activation(tf.sigmoid)
]
BSIZE = 50
NITER = 60000
T_SAMPLES = 10
P_SAMPLES = 50

CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?


def main():

    df_train, df_test = fetch_data()
    df = pd.concat((df_train, df_test))
    X_con, X_cat, Y = input_fn(df)
    Xt_con, Xs_con = np.split(X_con, [len(df_train)], axis=0)
    Xt_cat, Xs_cat = np.split(X_cat, [len(df_train)], axis=0)
    Yt, Ys = np.split(Y, [len(df_train)], axis=0)

    X_con_ = tf.placeholder(tf.float32, [None, Xt_con.shape[1]])
    X_cat_ = tf.placeholder(tf.float32, [None, Xt_cat.shape[1]])
    Y_ = tf.placeholder(tf.float32, [None, 1])
    N_ = tf.placeholder(tf.float32)

    # Make model
    features = [(X_con_, CON_LAYERS), (X_cat_, CAT_LAYERS)]
    likelihood = ab.bernoulli()
    Phi, loss = ab.featurenet(features, Y_, N_, LAYERS, likelihood, T_SAMPLES)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG):
        init.run()

        batches = ab.batch(
            {X_con_: Xt_con, X_cat_: Xt_cat, Y_: Yt},
            N_,
            batch_size=BSIZE,
            n_iter=NITER,
            seed=RSEED
        )
        for i, data in enumerate(batches):
            train.run(feed_dict=data)
            if i % 1000 == 0:
                loss_val = loss.eval(feed_dict=data)
                print("Iteration {}, loss = {}".format(i, loss_val))

        # Predict
        Eps = [Phi[0].eval(feed_dict={X_con_: Xs_con, X_cat_: Xs_cat})
               for _ in range(P_SAMPLES)]

    Ep = np.hstack(Eps).mean(axis=1)
    Ey = Ep > 0.5

    acc = accuracy_score(Ys.flatten(), Ey)
    logloss = log_loss(Ys.flatten(), np.stack((1 - Ep, Ep)).T)

    print("Accuracy = {}, log loss = {}".format(acc, logloss))


def fetch_data():
    train_file = tempfile.NamedTemporaryFile()
    test_file = tempfile.NamedTemporaryFile()
    req.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases"
                    "/adult/adult.data", train_file.name)
    req.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/"
                    "adult/adult.test", test_file.name)

    df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True,
                          skiprows=1)

    df_train[LABEL_COLUMN] = (df_train["income_bracket"]
                              .apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (df_test["income_bracket"]
                             .apply(lambda x: ">50K" in x)).astype(int)

    return df_train, df_test


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k)
    # to the values of that column stored in a constant Tensor.
    continuous_cols = [df[k].values for k in CONTINUOUS_COLUMNS]
    X_con = np.stack(continuous_cols).astype(np.float32).T

    # Standardise
    X_con -= X_con.mean(axis=0)
    X_con /= X_con.std(axis=0)

    # Creates a dictionary mapping from each categorical feature column name
    categorical_cols = [pd.get_dummies(df[k]).values for k in
                        CATEGORICAL_COLUMNS]
    X_cat = np.hstack(categorical_cols)

    # Converts the label column into a constant Tensor.
    label = df[LABEL_COLUMN].values[:, np.newaxis]

    # Returns the feature columns and the label.
    return X_con, X_cat, label


if __name__ == "__main__":
    main()
