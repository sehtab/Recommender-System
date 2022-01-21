# Recommender-System

This project is recommender system which will advise subscriber for movie rating. The key points of this project are:

* Collabortive Filtering

* Pearson Correlation

# Collaborative Filtering

Collaborative filtering is based on other customers choice which is kept in dataframe and similar group of people are provided with similar contents. User based collaborative system is based on user similarity or neighborhood.

# Algorithm

* User ratings matrix: shows the rating of users 
* Pearson Correlation: discover similarit of active user is to other users.Similarity measurents is done by Pearson Correlation
* Creating the weighted ratings matrix: correlation results in a weighted ratings matrix. Then matrix is normalized.
* Finding Similarity of users to input users
* Finding top similar users to input user
* Finding rating of selected users to all movies
* After sorting see the top 20 movies that algorithm recommended
