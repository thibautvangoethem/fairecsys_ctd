from trainRecommendor import  trainRecommendor
from createRecommendationMatrix import createRecommendMatrix
from calculateBER import calculateBER
from fairecsys import fairecsysmain

if __name__=="__main__":
    trainRecommendor()
    createRecommendMatrix()
    calculateBER()
    fairecsysmain()