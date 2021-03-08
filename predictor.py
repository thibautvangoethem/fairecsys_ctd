from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def BER(recommendation,clf,user_gender_list):
    '''
    Calculates the BER based on the average probability to make mistakes
    :param recommendation: recommendation matrix
    :param clf: the trained random forest classifier
    :param user_gender_list:
    :return:
    '''

    f_total=0 #amount of females predictions
    m_total=0  #amount of males predickions
    f_wrong=0 # predicted as female and is male
    m_wrong=0  # predicted as male and is female


    for index,rec in enumerate(recommendation):

        result = clf.predict([rec])[0]
        if  result=='M':
            m_total+=1
        else:
            f_total += 1
        if result != user_gender_list[index]:
            if user_gender_list[index] == 'M':
                f_wrong += 1
            else:
                m_wrong += 1

    return ((f_wrong/f_total)+(m_wrong/m_total))/2
