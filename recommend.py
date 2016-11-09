import pandas as pd
import numpy as np

movies_df = pd.read_csv('ml-1m/movies.dat',header=None,sep='::',names=['movie_id','movie_title','movie_genre'],engine='python')

movies_df = pd.concat([movies_df,movies_df.movie_genre.str.get_dummies(sep='|')],axis=1)

#print movies_df.head()

movie_categories = movies_df.columns[3:]

#print movies_df.loc[0]

from collections import OrderedDict

user_preferences = OrderedDict(zip(movie_categories,[]))

user_preferences['Action'] = 2
user_preferences['Adventure'] = 3
user_preferences['Animation'] = 1
user_preferences["Children's"] = 1
user_preferences["Comedy"] = 2
user_preferences['Crime'] = 3
user_preferences['Documentary'] = 2
user_preferences['Drama'] = 3
user_preferences['Fantasy'] = 2
user_preferences['Film-Noir'] = 1
user_preferences['Horror'] = 5
user_preferences['Musical'] = 1
user_preferences['Mystery'] = 5
user_preferences['Romance'] = 3
user_preferences['Sci-Fi'] = 2
user_preferences['War'] = 3
user_preferences['Thriller'] = 2
user_preferences['Western'] = 1

def get_movie_score(movie_features,user_preferences):
    return np.dot(movie_features,user_preferences)

sample_movie_features = movies_df.loc[648][movie_categories]

#print sample_movie_features

#user_predicted_score = np.dot(sample_movie_features,user_preferences.values())

#print user_predicted_score

#print movies_df[movie_categories].head(1)
#print user_preferences.values()

def get_movie_recommendation(user_preferences,number):
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score,args=([user_preferences.values()]),axis=1)
    return movies_df.sort_values(by=['score'],ascending=False)['movie_title'][:number]

def get_genre(user_preferences,number):
    movies_df['score'] = movies_df[movie_categories].apply(get_movie_score,args=([user_preferences.values()]),axis=1)
    return movies_df.sort_values(by=['score'],ascending=False)['movie_genre'][:number]

print get_movie_recommendation(user_preferences,10)
print get_genre(user_preferences,10)
