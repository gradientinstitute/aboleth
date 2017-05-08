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
CON_LAYERS = [
    ab.dense_var(output_dim=5, full=True, seed=RSEED)
]
LAYERS = [
    ab.random_arccosine(100, 1., seed=RSEED),
    ab.dense_var(output_dim=1, full=True, seed=RSEED),
    ab.activation(tf.sigmoid)
]
EMBED_DIMS = 3
BSIZE = 50
NITER = 60000
T_SAMPLES = 10
P_SAMPLES = 50

CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?


def main():

    # Get Continuous and categorical data
    df_train, df_test = fetch_data()
    df = pd.concat((df_train, df_test))
    X_con, X_cat, Y = input_fn(df)

    # Split data into training and testing
    Xt_con, Xs_con = np.split(X_con, [len(df_train)], axis=0)
    Xt_cat, Xs_cat = np.split(X_cat, [len(df_train)], axis=1)
    D_cat = len(X_cat)
    Yt, Ys = np.split(Y, [len(df_train)], axis=0)

    # Graph place holders
    X_con_ = tf.placeholder(tf.float32, [None, Xt_con.shape[1]])
    Y_ = tf.placeholder(tf.float32, [None, 1])
    N_ = tf.placeholder(tf.float32)

    # Create the categorical embedding inputs
    K = X_cat.max(axis=1).flatten() + 1
    D_out = [EMBED_DIMS] * D_cat
    X_cat_, catfeat = embedding_layers(D_out, K, full=False, seed=RSEED)

    # Feed dicts
    train_dict = {X_con_: Xt_con, Y_: Yt}
    train_dict.update(dict(zip(X_cat_, Xt_cat)))
    test_dict = {X_con_: Xs_con}
    test_dict.update(dict(zip(X_cat_, Xs_cat)))

    # Make model
    features = [(X_con_, CON_LAYERS)] + catfeat
    likelihood = ab.bernoulli()
    Phi, loss = ab.featurenet(features, Y_, N_, LAYERS, likelihood, T_SAMPLES)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    pred = ab.predict(Phi)
    init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG):
        init.run()

        batches = ab.batch(
            train_dict,
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
        Eps = [pred.eval(feed_dict=test_dict) for _ in range(P_SAMPLES)]

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
    categ_cols = [np.where(pd.get_dummies(df[k]).values)[1][:, np.newaxis]
                  for k in CATEGORICAL_COLUMNS]
    X_cat = np.array(categ_cols)

    # Converts the label column into a constant Tensor.
    label = df[LABEL_COLUMN].values[:, np.newaxis]

    # Returns the feature columns and the label.
    return X_con, X_cat, label


def embedding_layers(output_dims, n_categories, *args, **kwargs):

    X_, features = [], []
    for o, k in zip(output_dims, n_categories):
        x_ = tf.placeholder(tf.int32, [None, 1])
        layer = [ab.embedding_var(o, k, *args, **kwargs)]
        features.append((x_, layer))
        X_.append(x_)

    return X_, features


if __name__ == "__main__":
    main()
