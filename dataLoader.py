import csv
import pandas as pd
class data_loader():
    def __init__(self):
        self.userlist=pd.read_csv("ml-1m/users.dat",sep="::",names=["userid","gender","age","occupation","zip"])
        self.movies=pd.read_csv("ml-1m/movies.dat",sep="::",names=["movieid","name","genres"])
        self.ratings=pd.read_csv("ml-1m/ratings.dat",sep="::",names=["userid","movieid","rating","timestamp"])
        self.ratings=self.ratings[[int(x) > 3 for x in self.ratings["rating"]]]

        movies = len(self.movies)
        users = len(self.userlist)
        self.matrix = [[0]*movies]*users
        self.merged=pd.merge(self.ratings,self.userlist,on="userid")

        movie_list = self.movies.movieid.tolist()
        for i, row in self.merged.iterrows():
            self.matrix[row[0]-1][movie_list.index(row[1])] = 1
if __name__ == '__main__':
    loader=data_loader()