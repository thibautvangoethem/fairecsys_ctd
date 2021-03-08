import numpy as np
import pandas as pd
import pickle
from libreco.data import random_split, DatasetPure
from libreco.algorithms import SVDpp  # pure data, algorithm SVD++
from libreco.evaluation import evaluate

def createRecommendMatrix():
    '''
    Creates a recommendation matrix based on SVDpp that is trained on the movielens dataset
    :return:
    '''
    data = pd.read_csv("ml-1m/ratings.dat", sep="::",
                       names=["user", "item", "label", "time"])

    # split whole data into three folds for training, evaluating and testing
    train_data, eval_data, test_data = random_split(data, multi_ratios=[0.8, 0.1, 0.1])

    train_data, data_info = DatasetPure.build_trainset(train_data)

    svdpp=SVDpp.load("model","trained_matrix",data_info)

    recommendation_lists = []
    for i in range(1,6040):
        recommendation_lists.append(svdpp.recommend_user(user=i, n_rec=10))

    pickle.dump(recommendation_lists, open("data/rec_lists.dump", "wb"))


    rec_matrix = np.zeros((6040,3952))
    for index,item in enumerate(recommendation_lists):
        for recommendation in item:
            rec_matrix[index][recommendation[0]] = 1

    pickle.dump(rec_matrix,open("data/rec_matrix.dump","wb"))

if __name__ == '__main__':
    createRecommendMatrix()