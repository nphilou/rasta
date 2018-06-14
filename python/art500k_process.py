import pandas as pd
import numpy as np
import os
from os.path import join
import argparse
import matplotlib.pyplot as plt

PATH = os.path.dirname(__file__)


def get_subset(dataset, indices):
    return dataset[indices]


def split_labels(labels, sep=10000):
    return labels[:sep], labels[sep:]


if __name__ == '__main__':

    # LABEL_PATH = '/home/local/shared/bnegrevergne/data/art500k/data_labels_v1.txt'
    # /Users/Philippe/Programmation/rasta/
    LABEL_PATH = '../data/art500k/data_labels_v1.txt'

    df = pd.read_table(LABEL_PATH,
                       sep='\t|\s{4,}', header=0, engine='python')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #    print(df.loc[:, 'style'].dropna())

    count = df.groupby('style').size().sort_values(ascending=False)

    plt.rcParams["figure.figsize"] = (40, 10)
    plt.bar(count.keys(), count)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('classes.png')
    plt.show()

    # df.loc[:, 'style'].dropna().plot(x='style')
    # print(df.loc[df['style'].isin(['Impressionism', 'Expressionism'])].head())

    # print(df.loc[df['style'].isin(['Impressionism', 'Expressionism'])][['img_id', 'style']].head())

    # print(df['style'].isin(['Impressionism', 'Expressionism'])[262924:262935])

    keep_list = df['style'].isin(['Impressionism', 'Expressionism']).tolist()

    # keep_list = [True, True, False, True, False, False]

    batch_count = 37
    files_name = ["/home/local/shared/bnegrevergne/data/art500k/img_encoding_%d.npy" % i for i in range(batch_count)]

    # files_name = ['/Users/Philippe/Programmation/rasta/data/tmp/tmp.npy',
    #               '/Users/Philippe/Programmation/rasta/data/tmp/tmp2.npy']

    shape = [224, 224, 3]

    X_train = np.empty(shape=shape)

    for file in files_name:
        batch = np.load(file)
        keep_beg, keep_end = split_labels(keep_list, sep=10000)
        subset = get_subset(batch, keep_beg)
        X_train = np.append(X_train, subset, axis=0)
        keep_list = keep_end

    print(X_train.shape)
