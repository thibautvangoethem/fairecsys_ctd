import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import SVDpp  # pure data, algorithm SVD++
from libreco.evaluation import evaluate

def trainRecommendor():
    '''
    Trains the SVDpp on the movielens dataset
    :return:
    '''
    data = pd.read_csv("ml-1m/ratings.dat", sep="::",
                       names=["user", "item", "label", "time"])

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    test_data = DatasetPure.build_testset(test_data)
    print(data_info)

    svdpp = SVDpp(task="rating", data_info=data_info, embed_size=16, n_epochs=3, lr=0.001,
                  reg=None, batch_size=256)
    # monitor metrics on eval_data during training
    svdpp.fit(train_data, verbose=2, eval_data=eval_data, metrics=["rmse", "mae", "r2"])

    print("evaluate_result: ", evaluate(model=svdpp, data=test_data,
                                        metrics=["rmse", "mae"]))

    svdpp.save("model", "trained_matrix",inference_only=True)

if __name__ == '__main__':
    trainRecommendor()