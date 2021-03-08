import numpy as np
import pandas as pd
from libreco.data import random_split, DatasetPure
from libreco.algorithms import SVDpp  # pure data, algorithm SVD++
from libreco.evaluation import evaluate
def singlerecommend(user_id):
    data = pd.read_csv("ml-1m/ratings.dat", sep="::",
                       names=["user", "item", "label", "time"])

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, data_info = DatasetPure.build_trainset(train_data)

    svdpp=SVDpp.load("model","trained_matrix",data_info)

    val=svdpp.recommend_user(user=user_id, n_rec=10)
    print("recommendation: ",val )
    return val