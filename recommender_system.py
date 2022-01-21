#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:10:11 2021

@author: altair
"""

import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

print(movies_df.head())
print(ratings_df.head())

# using regular repression to find a year stored between parentheses
movies_df['year'] = movies_df.title.str.extract('(\d\d\d\d)',expand=False)

# removing the years from the 'title' column.
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# dropping genres column
movies_df = movies_df.drop('genres', 1)
print(movies_df.head())
print(ratings_df.head())

# collaborative filtering
userinput = [{'title': 'Breakfast Club, The', 'rating': 5},
             {'title': 'Toy Story', 'rating': 3.5},
             {'title': 'Jumanji', 'rating': 2},
             {'title': 'Akira', 'rating': 4.5}]
inputmovies = pd.DataFrame(userinput)
print(inputmovies)

# filtering out the movie by title
inputid = movies_df[movies_df['title'].isin(inputmovies['title'].tolist())]
inputmovies = pd.merge(inputid, inputmovies)
inputmovies = inputmovies.drop('year',1)
print(inputmovies)

# filtering out users that have watched movies that the input has watched and storing it
usersubset = ratings_df[ratings_df['movieId'].isin(inputmovies['movieId'].tolist())]
print(usersubset.head())
usersubsetgroup = usersubset.groupby(['userId'])
usersubsetgroup = sorted(usersubsetgroup, key = lambda x: len(x[1]),reverse=True)
#print(usersubsetgroup)

usersubsetgroup = usersubsetgroup[0:100]

# pearson correlation
pearsonCorrelationDict = {}
for name, group in usersubsetgroup:
    group = group.sort_values(by='movieId')
    nratings = len(group)
    temp_df = inputmovies[inputmovies['movieId'].isin(group['movieId'].tolist())]
    tempratinglist = temp_df['rating'].tolist()
    tempgrouplist = temp_df['rating'].tolist()
    sxx = sum([i**2 for i in tempratinglist])-pow(sum(tempratinglist),2)/float(nratings)
    syy = sum([i**2 for i in tempgrouplist])-pow(sum(tempgrouplist),2)/float(nratings)
    sxy = sum(i*j for i, j in zip(tempratinglist, tempgrouplist)) - sum(tempratinglist) * sum(tempgrouplist)/float(nratings)
    
    if sxx != 0 and syy != 0:
        pearsonCorrelationDict[name] = sxy/sqrt(sxx*syy)
    else:
        pearsonCorrelationDict[name] = 0
        
pearsondf = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsondf.columns = ['similarityIndex']
pearsondf['userId'] = pearsondf.index
pearsondf.index = range(len(pearsondf))
print(pearsondf.head())

topusers = pearsondf.sort_values(by='similarityIndex', ascending=False)[0:50]
print(topusers.head())

topusersrating = topusers.merge(ratings_df, left_on = 'userId', right_on = 'userId', how='inner')
print(topusersrating.head())
# multiplies the similarity by the user's ratings
topusersrating['weightRatings'] = topusersrating['similarityIndex']*topusersrating['rating']
print(topusersrating.head())

# applies a sum to the topusers after groupng it up by userid
temptopusersrating = topusersrating.groupby('movieId').sum()[['similarityIndex', 'weightRatings']]
temptopusersrating.columns = ['sum_similarityIndex', 'sum_weightedRating']
print(temptopusersrating.head())

# create an empty dataframe
recommendation_df = pd.DataFrame()

# now we take the weighted average
recommendation_df['weighted average recommendation score'] = temptopusersrating['sum_weightedRating']/temptopusersrating['sum_similarityIndex']
recommendation_df['movieId'] = temptopusersrating.index
print(recommendation_df.head())

# sorting and see top 20   movies that the algorithm recommended
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
print(recommendation_df.head(10)) 
    

