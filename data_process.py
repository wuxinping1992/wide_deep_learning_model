import os
from urllib import request
import pandas as pd
import tensorflow as tf


# maybe download data set

def download_data_set(train_data, test_data):
    if os.path.exists(train_data):
        pass
    else:
        request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data",
                            './data/adult.data')
        print("Training data is downloaded to %s" % train_data)

    if os.path.exists(test_data):
        pass
    else:
        request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
                            './data/adult.test')
        print("Testing data is downloaded to %s" % test_data)

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

LABEL_COLUMN = "label"

if __name__ == "__main__":
    download_data_set('./data/adult.data', './data/adult.test')
    df_train = pd.read_csv(tf.gfile.Open('./data/adult.data'), names=COLUMNS, skipinitialspace=True, engine="python")
    print(df_train[:10])
