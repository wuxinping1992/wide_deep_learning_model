import tensorflow as tf
from tensorflow.contrib import layers, learn
from data_process import download_data_set
import pandas as pd
import argparse
import sys


COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

# build model


def build_estimator(model_dir, model_type):
    """build an estimator"""

    # base sparse feature process
    gender = layers.sparse_column_with_keys(column_name='gender', keys=['female', 'male'])
    education = layers.sparse_column_with_hash_bucket(column_name='education', hash_bucket_size=1000)
    relationship = layers.sparse_column_with_hash_bucket(column_name='relationship', hash_bucket_size=100)
    workclass = layers.sparse_column_with_hash_bucket(column_name='workclass', hash_bucket_size=100)
    occupation = layers.sparse_column_with_hash_bucket(column_name='occupation', hash_bucket_size=1000)
    native_country = layers.sparse_column_with_hash_bucket(column_name='native_country', hash_bucket_size=1000)

    # base continuous feature
    age = layers.real_valued_column(column_name='age')
    education_num = layers.real_valued_column(column_name='education_num')
    capital_gain = layers.real_valued_column(column_name='capital_gain')
    capital_loss = layers.real_valued_column(column_name='capital_loss')
    hours_per_week = layers.real_valued_column(column_name='hours_per_week')

    # transformation.bucketization 将连续变量转化为类别标签。从而提高我们的准确性
    age_bucket = layers.bucketized_column(source_column=age,
                                          boundaries=[18, 25, 30, 35, 40, 45,50, 55, 60, 65])

    # wide columns and deep columns
    # 深度模型使用到的特征和广度模型使用到的特征
    # 广度模型特征只只用到了分类标签
    wide_columns = [gender, native_country, education, relationship, workclass, occupation, age_bucket,
                    layers.crossed_column(columns=[education, occupation], hash_bucket_size=int(1e4)),
                    layers.crossed_column(columns=[age_bucket, education, occupation], hash_bucket_size=int(1e6)),
                    layers.crossed_column(columns=[native_country, occupation], hash_bucket_size=int(1e4))]

    deep_columns = [layers.embedding_column(workclass, dimension=8),
                    layers.embedding_column(education, dimension=8),
                    layers.embedding_column(gender, dimension=8),
                    layers.embedding_column(relationship, dimension=8),
                    layers.embedding_column(native_country, dimension=8),
                    layers.embedding_column(occupation, dimension=8),
                    age, education_num, capital_gain, capital_loss, hours_per_week]

    if model_type == "wide":
        m=learn.LinearClassifier(feature_columns=wide_columns, model_dir=model_dir)
    elif model_type == "deep":
        m=learn.DNNClassifier(feature_columns=deep_columns, model_dir=model_dir, hidden_units=[100, 50])
    else:
        m=learn.DNNLinearCombinedClassifier(model_dir=model_dir,
                                            linear_feature_columns=wide_columns,
                                            dnn_feature_columns=deep_columns,
                                            dnn_hidden_units=[256, 128, 64],
                                            dnn_activation_fn=tf.nn.relu)
    return m


# 模型输入函数，对数据集进行处理

def input_fn(df):
    """input builder function"""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS
    }
    # merge the two dictionaries into one
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    print(feature_cols.keys())
    # converts the label column into a constanct Tensor
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data_path, test_data_path):
    download_data_set(train_data_path, test_data_path)
    df_train = pd.read_csv(tf.gfile.Open(train_data_path),
                           names=COLUMNS,
                           skipinitialspace=True,
                           engine="python")
    df_test = pd.read_csv(tf.gfile.Open(test_data_path),
                          names=COLUMNS,
                          skipinitialspace=True,
                          skiprows=1,
                          engine="python")

    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_train.dropna(how='any', axis=0)

    df_train[LABEL_COLUMN] = df_train["income_bracket"].apply(lambda x: '>50k' in x).astype(int)
    df_test[LABEL_COLUMN] = df_test["income_bracket"].apply(lambda x: '>50k' in x).astype(int)
    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    result = m.evaluate(input_fn=lambda :input_fn(df_test), steps=1)
    print(type(result))
    print(result)
    for key in sorted(result):
        print("%s: %s"%( key, result[key]))

FLAGS = None

def main(_):
    # train_and_eval(model_dir='./model/', model_type='', train_steps=200, train_data_path='./data/adult.data', test_data_path='./data/adult.test')
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps, FLAGS.train_data_path, FLAGS.test_data_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model/",
        help="Base directory for output models."

    )

    parser.add_argument(

        "--model_type",
        type=str,
        default="",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )

    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./data/adult.data",
        help="Path to the training data."
    )

    parser.add_argument(
        "--test_data_path",
        type=str,
        default="./data/adult.test",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)






