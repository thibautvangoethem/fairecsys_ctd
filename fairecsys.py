import pandas as pd
import pickle
import numpy as np
import math
import copy

from oauthlib.openid.connect.core.endpoints import userinfo

from predictor import BER
from sklearn.ensemble import RandomForestClassifier

test = set()


def createGenderList():
    '''
    Creates a list of all the genders for every user in the dataset
    :return:
    '''
    userlist = pd.read_csv("ml-1m/users.dat", sep="::", names=["userid", "gender", "age", "occupation", "zip"])
    genderlist = userlist.gender.values.tolist()
    return genderlist


def create_Contingency(matrix):
    '''
    Creates a contingency matrix based on the original recommendation matrix and the effective gender data
    :param matrix:
    :return:
    '''
    unique = np.unique(matrix, axis=0)
    len_unique = len(unique)
    H = np.zeros((2, len_unique))

    unique_normal = unique.tolist()
    unique_strings = []
    for row in unique_normal:
        unique_strings.append(str(row))

    gender = createGenderList()

    for index, item in enumerate(matrix):
        if gender[index] == "F":
            row = 1
        else:
            row = 0

        tmp_list = str(item.tolist())
        column = unique_strings.index(tmp_list)

        H[row][column] += 1

    return (H, unique_strings)


def movegender(C, b0, b1, eps, H_tuple, user_gender_list, gender=1):
    '''
    The MoveGender function that assigns multiple users a new recommendation matrix
    :param C: Original recommendation matrix
    :param b0: Amount of males
    :param b1: Amount of females
    :param eps: epsilon value for epsilon-fairness we want to reach
    :param H_tuple: tuple with contingency table and information for indexing this table
    :param user_gender_list:
    :param gender: What gender we are moving from
    :return:
    '''
    s_y = []
    Y0 = []
    Y1 = []
    H = H_tuple[0]
    H_values = H_tuple[1]

    C_tmp = C.tolist()
    C_string = []
    for item in C_tmp:
        C_string.append(str(item))

    for i in range(len(H_values)):
        tmp = (H[0][i] / b0) - (H[1][i] / b1)
        s_y.append(tmp)
        if tmp >= 0:
            Y0.append(H_values[i])
        else:
            Y1.append(H_values[i])

    sumY0 = 0
    for y in Y0:
        sumY0 += H[1][H_values.index(y)]
    sumY1 = 0
    for y in Y1:
        sumY1 += H[0][H_values.index(y)]
    delta = 1 / 2 * ((sumY0 / b1) + (sumY1 / b0))

    if (eps - delta) <= 0:
        return C, delta

    py = []
    for y in Y0:
        if gender == 0:
            py.append(math.floor(b0 * s_y[H_values.index(y)]))
        else:
            py.append(math.floor(b1 * s_y[H_values.index(y)]))

    qy = []
    for y in Y1:
        if gender == 0:
            qy.append(math.ceil(-b0 * s_y[H_values.index(y)] - 1))
        else:
            qy.append(math.ceil(-b1 * s_y[H_values.index(y)] - 1))
    py = np.array(py)
    qy = np.array(qy)

    t = min(py.sum(), qy.sum())
    if gender == 0:
        l = min(math.ceil(2 * b0 * (eps - delta)), t)
    else:
        l = min(math.ceil(2 * b1 * (eps - delta)), t)

    for i in range(l):
        if gender == 1:
            # from q to p for female
            from_female = np.where(qy >= 1)[0]
            to_male = np.where(py >= 1)[0]
            while (True):
                from_female_choice = np.random.randint(0, len(from_female))
                to_male_choice = np.random.randint(0, len(to_male))

                from_vector = Y1[from_female[from_female_choice]]
                to_vector = Y0[to_male[to_male_choice]]

                user1 = C_string.index(from_vector)
                if (user_gender_list[user1] == "M"):
                    qy[from_female[from_female_choice]] = 0
                    continue
                user2 = C_string.index(to_vector)
                C[user1] = copy.deepcopy(C[user2])

                qy[from_female[from_female_choice]] -= 1
                py[to_male[to_male_choice]] -= 1
                C_string[user1] = C_string[user2]
                break
        else:
            # from p to q for male
            to_female = np.where(qy >= 1)[0]
            from_male = np.where(py >= 1)[0]
            while (True):
                to_female_choice = np.random.randint(0, len(to_female))
                from_male_choice = np.random.randint(0, len(from_male))

                to_vector = Y1[to_female[to_female_choice]]
                from_vector = Y0[from_male[from_male_choice]]

                user1 = C_string.index(from_vector)
                if (user_gender_list[user1] == "F"):
                    py[from_male[from_male_choice]] = 0
                    continue
                user2 = C_string.index(to_vector)
                C[user1] = copy.deepcopy(C[user2])

                qy[to_female[to_female_choice]] -= 1
                py[from_male[from_male_choice]] -= 1
                C_string[user1] = C_string[user2]
                break

    if gender == 0:
        delta += l / (2 * b0)
    else:
        delta += l / (2 * b1)

    return C, delta


def fairecsys(C, b0, b1, eps, user_gender_list):
    '''
    The general fairecsys algorithm
    :param C: Original recommendation matrix
    :param b0: Amount of males
    :param b1: Amount of females
    :param eps: epsilon value for epsilon-fairness we want to reach
    :param user_gender_list:
    :return:
    '''
    H_tuple = create_Contingency(C)
    C, delta = movegender(C, b0, b1, eps, H_tuple, user_gender_list)
    if delta >= eps:
        return C, delta
    C, delta = movegender(C, b0, b1, eps, H_tuple, user_gender_list, 0)
    return C, delta

def fairecsysmain():
    '''
    Handles the complete algorithm and everything it needs including the creation of graphs to plot the result
    :return:
    '''
    results = list()
    matrix = pickle.load(open("data/rec_matrix.dump", "rb"))
    user_gender_list = createGenderList()
    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(matrix, user_gender_list)

    results.append(BER(matrix, clf, user_gender_list))

    b0 = 0
    b1 = 0

    for item in user_gender_list:
        if item == "F":
            b1 += 1
        else:
            b0 += 1

    for i in range(1, 9):
        temp = copy.deepcopy(matrix)
        C, delta = fairecsys(temp, b0, b1, i * 0.05, user_gender_list)
        results.append(BER(C, clf, user_gender_list))
        print("1 test done")
    print(results)

if __name__ == '__main__':
    results = list()
    matrix = pickle.load(open("data/rec_matrix.dump", "rb"))
    user_gender_list = createGenderList()
    clf=RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(matrix, user_gender_list)

    results.append(BER(matrix, clf, user_gender_list))

    b0 = 0
    b1 = 0

    for item in user_gender_list:
        if item == "F":
            b1 += 1
        else:
            b0 += 1

    for i in range(1, 9):
        temp=copy.deepcopy(matrix)
        C, delta = fairecsys(temp, b0, b1, i * 0.05, user_gender_list)
        results.append(BER(C, clf, user_gender_list))
        print("1 test done")
    print(results)
