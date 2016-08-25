import pandas as pd
import numpy as np

movies_df = pd.read_csv('ml-1m/movies.dat',header=None,sep='::',names=['movie_id','movie_title','movie_genre'],engine='python')

movies_df = pd.concat([movies_df,movies_df.movie_genre.str.get_dummies(sep='|')],axis=1)

print movies_df.head()
