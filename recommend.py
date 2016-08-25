import pandas as pd
import numpy as np

movies_df = pd.read_csv('ml-1m/movies.dat',header=None,sep='::',names=['movie_id','movie_title','movie_genre'],engine='python')

movies_df = pd.concat([movies_df,movies_df.movie_genre.str.get_dummies(sep='|')],axis=1)

#print movies_df.head()

movie_categories = movies_df.columns[3:]

#print movies_df.loc[0]

from collections import OrderedDict

user_preferences = OrderedDict(zip(movie_categories,[]))

user_preferences['Action'] = 5
user_preferences['Adventure'] = 5
user_preferences['Animation'] = 1
user_preferences["Children's"] = 1
user_preferences["Comedy"] = 3
user_preferences['Crime'] = 2
user_preferences['Documentary'] = 1  
user_preferences['Drama'] = 1
user_preferences['Fantasy'] = 5
user_preferences['Film-Noir'] = 1
user_preferences['Horror'] = 2
user_preferences['Musical'] = 1
user_preferences['Mystery'] = 3
user_preferences['Romance'] = 1
user_preferences['Sci-Fi'] = 5
user_preferences['War'] = 3
user_preferences['Thriller'] = 2
user_preferences['Western'] = 1
