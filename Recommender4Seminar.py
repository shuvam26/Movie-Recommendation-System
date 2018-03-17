import numpy as np
import math
from cmath import phase
from datetime import date
from inspect import getsourcefile
from os.path import abspath
from scipy.spatial.distance import cosine
import warnings
#import nimfa
import SlopeOne
import pickle
from sklearn.cluster import KMeans
import itertools
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from MatrixFactorization import matrixFactorization


warnings.filterwarnings("ignore")


class Recommender4Seminar:

    # self.columns          : columns of the ID table
    # self.columnsIDtoTITLE : a dictionary that maps the movie ID to TITLE
    # self.columnsTITLEtoID : dictionary that maps TITLE to ID

    # self.rows             : row of ID table
    # self.rowsROWtoID      : dictionary that maps ROW to ID
    # self.rowsIDtoROW      : dictionary that maps ID to ROW

    # self.movieData        :  table of total data
    # self.normalized       :  movieData table but Normalized
    # self.movieDataLearn   :  learn movie tables
    # self.movieDataTest    :  movie test table

    # self.user_avgs        : a dictionary that stores average user ratings
    # self.items_avgs       : a dictionary that stores average movie ratings

    # READING AND WRITING DATA
    # the method reads the original MovieLens data and copies them to the tables
    def loadAndSave_MovielensData(self):
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        moviesFile = open(absolute_path + "/data/movielens/movies.dat", "r", errors="ignore")
        ratingFile = open(absolute_path + "/data/movielens/user_ratedmovies.dat", "r", encoding='utf-8')
        self.movieDataLearn = np.zeros([2113, 10197])  # , dtype="int8")
        self.movieDataTest = np.zeros([2113, 10197])   # , dtype="int8")
        moviesFileLines = moviesFile.readlines()[1:]
        ratingFileLines = ratingFile.readlines()[1:]

        self.columns = []
        self.columnsIDtoTITLE = {}
        self.columnsTITLEtoID = {}

        self.rows = []
        self.rowsROWtoID = {}
        self.rowsIDtoROW = {}

        for m in range(len(moviesFileLines)):
            movie = moviesFileLines[m].split("\t")
            id = movie[0]
            title = movie[1]

            if id not in self.columns:
                self.columns.append(id)
                self.columnsIDtoTITLE[id] = title
                self.columnsTITLEtoID[title] = id
        self.columns = np.array(self.columns)

        counter = 0
        for r in range(len(ratingFileLines)):
            rating = ratingFileLines[r].split("\t")
            userID = rating[0]
            movieID = rating[1]
            score = rating[2]
            dateDay = rating[3]
            dateMonth = rating[4]
            dateYear = rating[5]

            if userID not in self.rowsIDtoROW:
                self.rows.append(counter)
                self.rowsROWtoID[counter] = userID
                self.rowsIDtoROW[userID] = counter
                counter += 1

            row = self.rowsIDtoROW[userID]
            movieColumn = np.where(self.columns == movieID)[0][0]
            rowDate = date(int(dateYear), int(dateMonth), int(dateDay))
            theDate = date(2008, 6, 1)
            if (rowDate <= theDate):  # before 1.6.2008 put in learn data
                self.movieDataLearn[row, movieColumn] = score
            else:  # after 1.6.2008 put in test data
                self.movieDataTest[row, movieColumn] = score
        self.movieData = np.add(self.movieDataLearn, self.movieDataTest)
        return

    # The method saves the tables in the txt file
    def saveDataToFile(self):
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        np.savetxt(absolute_path + '/parsedData/Seminar/movieData.txt', self.movieData, delimiter=',', fmt="%i")
        np.savetxt(absolute_path + '/parsedData/Seminar/movieDataLearn.txt', self.movieDataLearn, delimiter=',', fmt="%i")
        np.savetxt(absolute_path + '/parsedData/Seminar/movieDataTest.txt', self.movieDataTest , delimiter=',', fmt="%i")
        np.savetxt(absolute_path + '/parsedData/Seminar/rows.txt', self.rows, delimiter=',', fmt="%s")
        np.savetxt(absolute_path + '/parsedData/Seminar/columns.txt', self.columns, delimiter=',', fmt="%s")
        columnIDtoTITLE = open(absolute_path + "/parsedData/Seminar/columnsIDtoTITLE.txt", "w")
        string = ""
        for (key, val) in self.columnsIDtoTITLE.items():
            string += str(key) + "," + str(val) + "\n"
        string = string[:-1]
        columnIDtoTITLE.write(string)
        string = ""
        rowIDtoROW = open(absolute_path + "/parsedData/Seminar/rowsIDtoROW.txt", "w")
        for (key, val) in self.rowsIDtoROW.items():
            string += str(key) + "," + str(val) + "\n"
        string = string[:-1]
        rowIDtoROW.write(string)
        return

    # the method reads the table from txt and loads them in the tables
    def readDataFromFile(self):
        self.columnsIDtoTITLE = {}
        self.columnsTITLEtoID = {}
        self.rowsIDtoROW = {}
        self.rowsROWtoID = {}
        self.user_avgs = {}
        self.item_avgs = {}
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]

        movieData = open(absolute_path + '/parsedData/Seminar/movieData.txt', "r")
        self.movieData = np.loadtxt(movieData, delimiter=',')

        movieDataTest = open(absolute_path + '/parsedData/Seminar/movieDataTest.txt', "r")
        self.movieDataTest  = np.loadtxt(movieDataTest, delimiter=',')

        movieDataLearn = open(absolute_path + '/parsedData/Seminar/movieDataLearn.txt', "r")
        self.movieDataLearn  = np.loadtxt(movieDataLearn, delimiter=',')

        self.rows = np.genfromtxt(absolute_path + '/parsedData/Seminar/rows.txt', delimiter="\n", dtype='str')
        self.rows = np.array(self.rows)

        self.columns = np.genfromtxt(absolute_path + '/parsedData/Seminar/columns.txt', delimiter="\n", dtype='str')
        self.columns = np.array(self.columns)

        columnIDtoTITLE = open(absolute_path + "/parsedData/Seminar/columnsIDtoTITLE.txt", "r").readlines()
        rowIDtoROW = open(absolute_path + "/parsedData/Seminar/rowsIDtoROW.txt", "r").readlines()

        for l in columnIDtoTITLE:
            line = l.split(",")
            id = line[0].strip()
            title = "".join(line[1:]).strip()
            self.columnsIDtoTITLE[id] = title
            self.columnsTITLEtoID[title] = id
        for l in rowIDtoROW:
            line = l.split(",")
            id  = line[0].strip()
            row = line[1].strip()
            self.rowsIDtoROW[id] = row
            self.rowsROWtoID[row] = id

        item_avgs = open(absolute_path + "/parsedData/Seminar/item_avgs.txt", "r").readlines()
        user_avgs = open(absolute_path + "/parsedData/Seminar/user_avgs.txt", "r").readlines()

        for ia in item_avgs:
            line = ia.split(",")
            id = line[0].strip()
            score = line[1].strip()
            self.item_avgs[id] = score

        for ua in user_avgs:
            line = ua.split(",")
            self.user_avgs[line[0]] = line[1].strip()

        mask = np.ma.masked_where(self.movieData == 0, self.movieData)
        mean = np.mean(mask, axis=1)
        self.normalized = np.transpose(np.subtract(np.transpose(mask), mean))
        return

    # The method calculates the average of each user and each movie
    def compute_averages(self, m=0):
        #  formula = (vsota_ocen + m * global_avg) / (st.ocen + m)
        self.user_avgs = {}
        self.item_avgs = {}
        all_ratings = self.movieData[np.where(self.movieData != 0)]
        global_avg = sum(all_ratings) * 1.0 / len(all_ratings)
        for u in range(len(self.rows)):
            user = self.rows[u]
            sumRatings = sum(self.movieData[u,:])
            numRatings = np.count_nonzero(self.movieData[u,:])
            self.user_avgs[user] = (sumRatings + m * global_avg) / (numRatings + m)
        for m in range(len(self.columns)):
            movie = self.columns[m]
            sumRatings = sum(self.movieData[:,m])
            numRatings = np.count_nonzero(self.movieData[:,m])
            self.item_avgs[movie] = (sumRatings + m * global_avg) / (numRatings + m)
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]

        string = ""
        user_avgs = open(absolute_path + "/parsedData/Seminar/user_avgs.txt", "w")
        for (key, val) in self.user_avgs.items():
            string += str(key) + "," + str(val)  + "\n"
        string = string[:-1]
        user_avgs.write(string)

        string = ""
        item_avgs = open(absolute_path + "/parsedData/Seminar/item_avgs.txt", "w")
        for (key, val) in self.item_avgs.items():
            string += str(key) + "," + str(val) + "," + str(self.columnsIDtoTITLE[key]) + "\n"
        string = string[:-1]
        item_avgs.write(string)
        return

    # 1 : tell me which movie to watch

    # calculating the similarity of two products by cosine similarity
    def item_similarity(self, item1, item2):
        indeks1 = np.where(self.columns == item1)[0][0]
        indeks2 = np.where(self.columns == item2)[0][0]
        ocene1 = np.where(self.movieData[:,indeks1] != 0)
        ocene2 = np.where(self.movieData[:,indeks2] != 0)
        vse_ocene = np.intersect1d(ocene1, ocene2)
        item1 = self.normalized[vse_ocene, indeks1]
        item2 = self.normalized[vse_ocene, indeks2]
        try:
            return 1 - cosine(item1, item2)
        except:
            return 0

    # The method recommends n films
    def itemBased_recommendation(sely, movies, n):
        movies = np.array(movies)
        ocene = movies[:, 0].astype(int)
        filmi = movies[:, 1].astype(str)
        povprecje = np.mean(movies[:, 0].astype(int))
        indeksi = np.where(ocene > povprecje)
        ljubsi_filmi = filmi[indeksi]
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        file = open(absolute_path + "/output/Seminar/item_similarities.txt", "r").readlines() # for OSX
        result = []
        for line in file:
            movie1 = line.split(",")[1]
            movie2 = line.split(",")[3]
            similarity = line.split(",")[4].strip()
            if movie1 in ljubsi_filmi:
                result.append((movie2, similarity))
            elif movie2 in ljubsi_filmi:
                result.append((movie1, similarity))
            if len(set(result)) >= n:
                break
        sort = sorted(result, key=lambda tuple: tuple[1], reverse=True)
        return sort

    # calculations and records of the similarity of all pairs of films in terms of cosine similarity
    def most_similar_movies(self, n):
        all_combinations = itertools.combinations(self.columns, 2)
        combinations = []
        stevec = 0
        for ac in all_combinations:
            item1 = ac[0]
            item2 = ac[1]
            item1column = np.where(self.columns == item1)[0][0]
            item2column = np.where(self.columns == item2)[0][0]
            ocene1 = np.where(self.movieData[:, item1column] != 0)
            ocene2 = np.where(self.movieData[:, item2column] != 0)
            presek = np.intersect1d(ocene1, ocene2)
            if(len(presek) >= n):
                combinations.append(ac)
            print("prva zanka ", str(stevec*1.0 / 51984306))
            stevec += 1
        results = []
        stevec = 0
        for c in combinations:
            item1 = c[0]
            item2 = c[1]
            similarity = 0
            try:
                similarity = self.item_similarity(item1, item2)
            except:
                print("NAPAKA!")
                pass
            results.append((item1, self.columnsIDtoTITLE[item1], item2, self.columnsIDtoTITLE[item2], similarity))
            print(str(stevec) + "/" + str(len(combinations)), str(self.columnsIDtoTITLE[item1]), str(self.columnsIDtoTITLE[item2]), str(str(similarity)))
            stevec += 1
        sort = sorted(results, key = lambda tuple: tuple[4], reverse = True)
        print("pisem v datoteko")
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        #file = open(absolute_path + "/output/Seminar/item_similarities.txt", "w") # for OSX
        file = open(absolute_path + "\output\Seminar\item_similarities2.txt", "w")  # for Windows
        string = ""
        for tuple in sort:
            string += str(tuple[0]) + "," + str(tuple[1]) + "," + str(tuple[2]) + "," + str(tuple[3]) + "," + str(tuple[4]) + "\n"
        string = string[:-1]
        file.write(string)
        return

    # 2: evaluation

    # the method returns the list with all the accuracy calculations of the model
    def evaluate(self, data, predictions, binaryPredictions, binaryData):
        mae = mean_absolute_error(data, predictions)
        rmse = mean_squared_error(data, predictions)
        f1score = f1_score(binaryData, binaryPredictions)
        recall = recall_score(binaryData, binaryPredictions)
        precision = precision_score(binaryData, binaryPredictions)
        return [mae, rmse, recall, precision, f1score]

    # the method returns similarity to Pearson's correlation coefficient
    def user_similarity(self, user1, user2):
        indeks1 = np.where(self.rows == user1)[0][0]
        indeks2 = np.where(self.rows == user2)[0][0]
        ocene1 = np.where(self.movieData[indeks1,:] != 0)
        ocene2 = np.where(self.movieData[indeks2,:] != 0)
        vse_ocene = np.intersect1d(ocene1, ocene2)
        user1f = self.movieData[indeks1, vse_ocene]
        user2f = self.movieData[indeks2, vse_ocene]
        return pearsonr(user1f, user2f)[0]

    # calculations and records of the similarity of all pairs of users per pearson volume
    def most_similar_users(self, filename):
        combinations = []
        all_combinations = itertools.combinations(self.rows, 2)
        stevec = 0
        for ac in all_combinations:
            print(stevec , "/2231328")
            stevec += 1
            user1 = ac[0]
            user2 = ac[1]
            similarity = self.user_similarity(user1, user2)
            if not math.isnan(similarity):
                combinations.append((user1, user2, similarity))
        sort = sorted(combinations, key = lambda tuple: tuple[2], reverse = True)
        # record to file
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        file = open(absolute_path + filename, "w")
        string = ""
        for row in sort:
            string += str(self.rowsROWtoID[row[0]]) + "," + str(self.rowsROWtoID[row[1]]) + "," + str(row[2]) + "\n"
        string = string[:-1] + "\n"
        file.write(string)
        return

    # the method is similar to the above only for the first user to receive a vector of ratings, not an ID
    def user_similarity2(self, user1, user2):
        indeks2 = np.where(self.rows == user2)[0][0]
        ocene1 = np.where(user1 != 0)
        ocene2 = np.where(self.movieData[indeks2,:] != 0)
        vse_ocene = np.intersect1d(ocene1, ocene2)
        user1f = user1[vse_ocene]
        user2f = self.movieData[indeks2, vse_ocene]
        return pearsonr(user1f, user2f)[0]

    # the method recommends n films according to other users
    def userBased_recommendation(self, movies, n):
        user_scores = [0] * len(self.columns)
        for mov in movies:
            id = self.columnsTITLEtoID[mov[1]]
            indeks = np.where(self.columns == id)[0]
            user_scores[indeks] = int(mov[0])
        user_scores = np.array(user_scores)
        gledani = np.where(user_scores != 0)[0]

        result = [] # ID, ROW, SCORE
        for u in self.rows:
            similarity = self.user_similarity2(user_scores, u)
            if not math.isnan(similarity):
                result.append((self.rowsROWtoID[u], u, similarity))
        sort = sorted(result, key=lambda tuple: tuple[2], reverse=True)

        result = []
        for user in sort:
            row = user[1]
            user_scores = self.movieData[row, :]
            user_average = float(self.user_avgs[row])
            watched = np.where(user_scores >= user_average)[0]
            predlagaj = np.setdiff1d(watched, gledani)
            for mm in predlagaj:
                title = self.columnsIDtoTITLE[self.columns[mm]]
                if title not in result:
                    result.append(title)
        return result[:n]

    # the method creates dictionaries for SlopeOne
    def dataIntoLearnDictionarySaveTXT(self):

        n = 1000
        self.rows2 = self.rows[:n]
        self.columns2 = self.columns[:n]
        self.movieDataLearn = self.movieDataLearn[:n, :n]
        self.movieDataTest = self.movieDataTest[:n, :n]

        users = {}
        for u in range(len(self.rows2)):
            user = {}
            rated = np.where(self.movieDataLearn[u,:] != 0)[0]
            for r in rated:
                user[str(self.columnsIDtoTITLE[self.columns[r]])] = int(self.movieDataLearn[u, r])
            users[u] = user

        items = {}
        for i in range(len(self.columns2)):
            movie = {}
            rated = np.where(self.movieDataLearn[:, i] != 0)[0]
            title = str(self.columnsIDtoTITLE[self.columns2[i]])
            for r in rated:
                movie[int(r)]= int(self.movieDataLearn[r, i])
            items[title] = movie

        averages = self.buildAverageDiffs(items, users)

        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        with open(absolute_path + '/parsedData/Seminar/usersDictionary.pickle', 'wb') as f:
            pickle.dump(users, f)
        with open(absolute_path + '/parsedData/Seminar/itemsDictionary.pickle', 'wb') as f:
            pickle.dump(items, f)
        with open(absolute_path + '/parsedData/Seminar/averagesDictionary.pickle', 'wb') as f:
            pickle.dump(averages, f)
        return (users, items)

    # method dictionary assignments for SlopeOne
    def loadDictionaryData(self):
        n = 1000
        self.rows2 = self.rows[:n]
        self.columns2 = self.columns[:n]
        self.movieDataTest = self.movieDataTest[:n, :n]
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        with open(absolute_path + '/parsedData/Seminar/usersDictionary.pickle', 'rb') as f:
            self.usersDictionary = pickle.load(f)
        with open(absolute_path + '/parsedData/Seminar/itemsDictionary.pickle', 'rb') as f:
            self.itemsDictionary = pickle.load(f)
        with open(absolute_path + '/parsedData/Seminar/averagesDictionary.pickle', 'rb') as f:
            self.averagesDictionary = pickle.load(f)
        return

    # The method builds averages for SlopeOne
    def buildAverageDiffs(self, items, users):
        averages = {}
        for itemId in items.keys():
            for otherItemId in items.keys():
                average = 0
                userRatingPairCount = 0
                if itemId != otherItemId:
                    for userId in users:
                        userRatings = users[userId]
                        if itemId in userRatings and otherItemId in userRatings:
                            userRatingPairCount += 1
                            average += (userRatings[itemId] - userRatings[otherItemId])
                    if userRatingPairCount != 0:
                        averages[(itemId,otherItemId)] = average / userRatingPairCount
        return averages

    # the method compares the different methods and returns estimates RMSE, MAE, PRECISION, RECALL, F1
    def evaluating_different_methods(self):

        testSetMovies = np.where(self.movieDataTest != 0)
        realValueVector = []
        avgItemVector   = []
        avgUserVector   = []
        slopeOneVector  = []

        didSheLikeItAvg = []
        didSheLikeItS1 = []
        didSheReallyLikeIt = []

        s1 = SlopeOne.SlopeOne()

        bann_list = []

        for p in range(len(testSetMovies[0])):
            point = (testSetMovies[0][p], testSetMovies[1][p])

            if point[0] not in bann_list:
                realValue = self.movieDataTest[point[0], point[1]]

                userRatings = self.movieDataLearn[point[0], :]
                userRatings = userRatings[userRatings != 0]
                userAvg = np.mean(userRatings)

                movieRates = self.movieDataLearn[:, point[1]]
                movieRates = movieRates[movieRates != 0]
                itemAvg = np.mean(movieRates)

                title = self.columnsIDtoTITLE[self.columns2[point[1]]]
                try:
                    slopeOne = s1.suggestedRating(self.usersDictionary, self.itemsDictionary, self.averagesDictionary, point[0], title)
                except:
                    slopeOne = 0

                if slopeOne == 0.0:
                    bann_list.append(point[0])
                else:
                    realValueVector.append(realValue)
                    avgItemVector.append(itemAvg)
                    avgUserVector.append(userAvg)
                    slopeOneVector.append(slopeOne)


                    if slopeOne >= userAvg:
                        didSheLikeItS1.append(True)
                    else:
                        didSheLikeItS1.append(False)

                    if itemAvg >= userAvg:
                        didSheLikeItAvg.append(True)
                    else:
                        didSheLikeItAvg.append(False)

                    if realValue >= userAvg:
                        didSheReallyLikeIt.append(True)
                    else:
                        didSheReallyLikeIt.append(False)

                    if len(slopeOneVector) > 500:
                        break

        return [self.evaluate(avgItemVector, realValueVector, didSheLikeItAvg, didSheReallyLikeIt),
                self.evaluate(slopeOneVector, realValueVector, didSheLikeItS1, didSheReallyLikeIt)]

    # 3: binding rules

    # method returns the txt file
    def save_basket_data(self, filename):
        basket = []
        self.movieData = self.movieData.astype(np.int32)
        for u in range(len(self.rows)):
            user = self.rows[u]
            user_avg = self.user_avgs[user]
            movie_index = np.where(self.movieData[u, :] > float(user_avg))[0]
            basket.append(self.columns[movie_index].tolist())
        basket = np.array(basket)
        # write to file
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        file = open(absolute_path + filename, "w")
        for row in basket:
            vrstica = ""
            for item in row:
                vrstica += self.columnsIDtoTITLE[item] +","
            vrstica = vrstica[:-1] + "\n"
            file.write(vrstica)
        return True

    # 4: controversial films and matrix factorization

    # the method returns the txt file to the most controversial films
    def controversialMovies(self, filename):
        cont_list = []
        for m in range(len(self.columns)):
            movie = self.columns[m]
            title = self.columnsIDtoTITLE[movie]
            index = np.nonzero(self.movieData[:,m])
            ratings = self.movieData[index, m][0]
            firstHalf  = ratings[:int(len(ratings)/2)]
            secondHalf = ratings[int(len(ratings)/2):]
            if len(secondHalf) != 0 and len(firstHalf) != 0:
                cont_ratio = sum(secondHalf)/len(secondHalf) - sum(firstHalf) / len(firstHalf)
                cont_list.append((title, abs(cont_ratio)))
        cont_list = np.array(cont_list)
        sort = sorted(cont_list, key = lambda tuple: tuple[1], reverse = True)
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        file = open(absolute_path + filename, "w")
        string = ""
        for row in sort:
            string += str(row[0]) + "," + str(row[1]) + "\n"
        string = string[:-1]
        file.write(string)
        return sort

    # the method returns the same file as the cosine distance matrix matrix
    def save_item_distance_data(self, filename):
        distance = np.array([[None] * len(self.columns)] * len(self.columns))
        for i in range(len(self.columns)):
            for j in range(i, len(self.columns)):
                item1 = self.columns[i]
                item2 = self.columns[j]
                dis12 = 1 - self.item_similarity(str(item1), str(item2))
                distance[i,j] = round(dis12, 4)
        distance = np.transpose(distance)
        # record to file
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        file = open(file + filename, "w")
        string = str(len(self.columns)) + " labeled\n"
        for row in range(len(distance)):
            vrstica = self.columns[row] + "\t"
            for item in distance[row]:
                if item != None:
                    vrstica += str(item) + "\t"
            vrstica = vrstica[:-1] + "\n"
            string += vrstica
        file.write(string)
        return True

    def kmeans(self):
        absolute_path = abspath(getsourcefile(lambda: 0))[:-23]
        genreFile = open(absolute_path + "/data/movielens/movie_genres.dat", "r", encoding='utf-8').readlines()[1:]
        countClasses = {}
        ids = []
        for line in genreFile:
            id = int(line.split("\t")[0].strip())
            genre = line.split("\t")[1].strip()
            if genre not in countClasses:
                if id not in ids:
                    countClasses[genre] = 1
                    ids.append(id)
            else:
                if id not in ids:
                    countClasses[genre] += 1


        real = sorted([(x, countClasses[x]) for x in countClasses.keys()], key= lambda tuple : tuple[1], reverse=True)

        classes = len(countClasses.keys())
        ucna = self.movieData.T
        razredi = [[] for i in range(classes)]

        kmeans = KMeans(n_clusters=classes)
        napovedi = kmeans.fit_predict(ucna)

        for tocka, predvideno in zip(ucna, napovedi):
            razredi[predvideno].append(tocka)
        konc = sorted([len(x) for x in razredi], reverse=True)

        return (real, konc)


    def matrixFactorization(self):
        matrix = self.movieData[:500, :500]
        """
        dot = matrix.dot(matrix.T)
        W,  H  = matrixFactorization(matrix, rank=5, max_iter=750)
        e1 = np.linalg.norm(dot - W.dot(H), ord="fro")**2
        print("Error1 ", e1)
        print("complex ", complex(e1))
        return phase(complex(e1))
        """

        lsnmf = nimfa.Lsnmf(matrix, seed='random_vcol', rank=2, max_iter=100)
        lsnmf_fit = lsnmf()
        print('Rss: %5.4f' % lsnmf_fit.fit.rss())
        print('Evar: %5.4f' % lsnmf_fit.fit.evar())
        print('K-L divergence: %5.4f' % lsnmf_fit.distance(metric='kl'))
        print('Sparseness, W: %5.4f, H: %5.4f' % lsnmf_fit.fit.sparseness())
        return [lsnmf_fit.fit.rss(), lsnmf_fit.fit.evar(), lsnmf_fit.distance(metric='kl'), lsnmf_fit.fit.sparseness()]


