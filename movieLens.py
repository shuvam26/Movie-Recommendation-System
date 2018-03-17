import Recommender4Seminar

r = Recommender4Seminar.Recommender4Seminar()

#r.loadAndSave_MovielensData()
#r.saveDataToFile()
#r.compute_averages()

r.readDataFromFile()

#r.dataIntoLearnDictionarySaveTXT()
#r.loadDictionaryData()


# FIRST STEP : Which movie to watch ?

#r.most_similar_movies(10)

movies = [[5, "The Matrix"], [4, "Gothika"], [5, "Batman"], [5, "Pulp Fiction"], [5, "The Godfather: Part II"], [5, "The Lord of the Rings: The Two Towers"],
          [2, "The Greatest Story Ever Told"], [3, "Proof of Life"], [2, "Halloween II"], [1, "Twelve Monkeys"], [3, "Waiting to Exhale"], [4, "The Guardian"],
          [1, "The Flintstones in Viva Rock Vegas"], [2, "Showtime"], [1, "The Cowboy Way"], [3, "Friday the 13th: A New Beginning"],[4, "Mission to Mars"],
          [4, "Against All Odds"], [3, "12 Angry Men"], [2, "Crossfire"]]
#item_recommended_movies = r.itemBased_recommendation(movies, 15)
#print("Item Recommended Movies: ", item_recommended_movies)
#print("\n")



# SECOND STEP : Evaluation

#r.most_similar_users("/output/Seminar/user_similarities.txt")

movies = [[4, "Dumb & Dumber"],[5, "American Pie"],[5, "A Fish Called Wanda"],[4, "MASH"],[5, "Life of Brian"],[4, "The Matrix"],[2, "WALL�E"],[4, "Men in Black"],
          [2, "A Beautiful Mind"],[3, "Star Trek: Nemesis"],[4, "The Godfather"], [3, "Pirates of the Caribbean: The Curse of the Black Pearl"], [2, "Sex and the City"],
          [5, "Pulp Fiction"], [1, "Alien"], [2, "Rocky"]]
#user_recommended_movies = r.userBased_recommendation(movies, 15)
#print("User Recommended Movies: ", user_recommended_movies)
#print("\n")

#grades = r.evaluating_different_methods()
#print("Evaluation:")
#print("Item Average : [mae, rmse, recall, precision, f1score] ", grades[0])
# [0.64492695964793545, 0.70010787662161977, 0.58454106280193241, 0.63350785340314131, 0.60804020100502509]
#print("SlopeOne     : [mae, rmse, recall, precision, f1score] ", grades[1])
# [0.66037890334565952, 0.6755781348903952, 0.62318840579710144, 0.66494845360824739, 0.64339152119700749]
#print("\n")

# THIRD STEP : Binding Rules

#r.save_basket_data("/output/Seminar/seminarMovies.basket")

# FOURTH STEP : Controversial Films, Matrix Factorization

# This method records movie file according to their controversy
# #r.controversialMovies("/output/Seminar/controversialMovies.txt")

# Method for matrix of distances
#r.save_item_distance_data("/output/Seminar/movieDistanceMatrix.dst")

# Matrix Factorization
err = r.matrixFactorization()
print("Matrix Factorization: ", err)

# algorithm kmeans above the learning matrix
#kmeans = r.kmeans()
#print("K means: ", kmeans) 
#How many superb, subversive films are in the crowd: [['Predictions: ', [670, 1443]],
# ['Real Above Average: 1184', 'Really submissive 929']]


# How many movies are there in a certain genre?
# ([('Drama', 5075), ('Comedy', 3565), ('Thriller', 1661), ('Romance', 1640), ('Action', 1445), ('Crime', 1085), ('Adventure', 1003), ('Horror', 978), ('Sci-Fi', 739), ('Fantasy', 533), ('Children', 517), ('Mystery', 495), ('War', 494), ('Documentary', 430), ('Musical', 421), ('Animation', 278), ('Western', 261), ('Film-Noir', 145), ('IMAX', 24), ('Short', 1)],
# [6506, 1040, 992, 411, 313, 221, 161, 135, 88, 52, 49, 42, 40, 38, 34, 32, 20, 11, 10, 2])
