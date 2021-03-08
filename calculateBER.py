import pickle
from sklearn.ensemble import RandomForestClassifier
from predictor import BER
import pandas as pd

def createGenderList():
    '''
    Creates a list of all the genders for every user in the dataset
    :return:
    '''
    userlist = pd.read_csv("ml-1m/users.dat", sep="::", names=["userid", "gender", "age", "occupation", "zip"])
    genderlist = userlist.gender.values.tolist()
    return genderlist

def calculateBER():
    '''
    Trains a random forest classifier based on the gender list and calls a BER calculation function
    :return:
    '''
    matrix = pickle.load(open("data/rec_matrix.dump", "rb"))
    user_gender_list = createGenderList()
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(matrix, user_gender_list)
    result = BER(matrix,clf,user_gender_list)
    print(result)

if __name__ == '__main__':
    calculateBER()