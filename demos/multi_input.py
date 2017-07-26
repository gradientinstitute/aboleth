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
ab.set_hyperseed(RSEED)

# Sample width of net
T_SAMPLES = 10
EMBED_DIMS = 3

BSIZE = 50
NITER = 60000
P_SAMPLES = 5  # results in T_SAMPLES * P_SAMPLES predcitions

CONFIG = tf.ConfigProto(device_count={'GPU': 0})  # Use GPU ?


def main():

    # Get Continuous and categorical data
    df_train, df_test = fetch_data()
    df = pd.concat((df_train, df_test))
    X_con, X_cat, n_cats, Y = input_fn(df)

    # Define our graph
    con_layer = ab.stack(ab.sample(T_SAMPLES),
                         ab.dense_var(output_dim=5, full=True))

    # Note every embed_var call can be different
    cat_layer_list = [ab.embed_var(i, EMBED_DIMS) for i in n_cats]

    # # Concatenate assuming each layer gets a single slice
    cat_layer = ab.stack(ab.sample(T_SAMPLES),
                         ab.slicecat(*cat_layer_list))

    net = ab.stack(ab.concat(con_layer, cat_layer),
                   ab.random_arccosine(100, 1.),
                   ab.dense_var(output_dim=1, full=True),
                   ab.activation(tf.sigmoid))

    # Split data into training and testing
    Xt_con, Xs_con = np.split(X_con, [len(df_train)], axis=0)
    Xt_cat, Xs_cat = np.split(X_cat, [len(df_train)], axis=0)
    Yt, Ys = np.split(Y, [len(df_train)], axis=0)

    # Graph place holders
    X_con_ = tf.placeholder(tf.float32, [None, Xt_con.shape[1]])
    X_cat_ = tf.placeholder(tf.int32, [None, Xt_cat.shape[1]])
    Y_ = tf.placeholder(tf.float32, [None, 1])

    # # Feed dicts
    train_dict = {X_con_: Xt_con, X_cat_: Xt_cat, Y_: Yt}
    test_dict = {X_con_: Xs_con, X_cat_: Xs_cat}

    # Make model
    N = len(Xt_con)
    likelihood = ab.bernoulli()
    Phi, kl = net(X_con_, X_cat_)

    loss = ab.elbo(Phi, Y_, N, kl, likelihood)
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session(config=CONFIG):
        init.run()

        batches = ab.batch(
            train_dict,
            batch_size=BSIZE,
            n_iter=NITER)
        for i, data in enumerate(batches):
            train.run(feed_dict=data)
            if i % 1000 == 0:
                loss_val = loss.eval(feed_dict=data)
                print("Iteration {}, loss = {}".format(i, loss_val))

        # Predict
        Ep = ab.predict_expected(Phi, test_dict, P_SAMPLES)

    Ey = Ep > 0.5

    acc = accuracy_score(Ys.flatten(), Ey.flatten())
    logloss = log_loss(Ys.flatten(), np.hstack((1 - Ep, Ep)))

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
    n_values = [np.amax(c) + 1 for c in categ_cols]
    X_cat = np.concatenate(categ_cols, axis=1).astype(np.int32)

    # Converts the label column into a constant Tensor.
    label = df[LABEL_COLUMN].values[:, np.newaxis]

    # Returns the feature columns and the label.
    return X_con, X_cat, n_values, label


if __name__ == "__main__":
    main()
