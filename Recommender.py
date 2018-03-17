import math
import random
from inspect import getsourcefile
from os.path import abspath
import codecs
import unicodedata
import warnings
import collections
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")


class Recommender:

    # items: all movies
    # user_ratings: movie information

    # user_avgs: average user ratings
    # item_avgs: average movie ratings

    # Numpy Array
    # rows: rows with user names
    # columns: columns with movie names
    # movieData: numpy array with data

    # movieIDvsTitleDict: a movie ID in his name

    # 1. 

    # Method for preparation of data in items in user_ratings
    def parse_database(self, file_name):
        self.items = []
        self.user_ratings = collections.defaultdict(dict)
        mode = 'none'
        data = open(file_name, 'r')
        for line in data:
            ln = line.strip()
            if not ln or ln[0] == '%': continue    # empty line or comment
            if ln == '[items]':
                # switch to parsing item data
                mode = 'items'
                continue
            if ln == '[users]':
                # switch to parsing user/rating data
                mode = 'users'
                iCount = len(self.items)
                continue
            if mode == 'items':
                self.items.append(ln)
            elif mode == 'users':
                ln = ln.split(',')
                # store as a 2d dictionary, leave zeros to use less memory
                for mi, rating in enumerate(ln[1:]):
                    if rating.strip() != "?":
                        # store users ratings
                        self.user_ratings[ln[0]][self.items[mi]] = float(rating)
                if len(ln) != iCount+1:    # check DB consistency
                    print("User %s has invalid number of ratings (%d)." % (ln[0], len(self.ratings[ln[0]])))
            else:
                print('Strange line in database:')
                print(line)

    # Method for preparation of data in numpy array
    def toNumpy(self):
        self.rows = [];
        self.columns = [];
        for user in self.user_ratings:
            self.rows.append(user)
        for movie in self.items:
            self.columns.append(movie)
        self.movieData = np.zeros([len(self.rows), len(self.columns)], dtype='int8')
        self.rows = np.array(self.rows)
        self.columns = np.array(self.columns)
        for user in self.user_ratings:
            indeks_userja = np.where(self.rows == user)[0][0]
            for movie in self.user_ratings[user]:
                indeks_filma = np.where(self.columns == movie)[0][0]
                self.movieData[indeks_userja, indeks_filma] = self.user_ratings[user][movie]
        return

    # This method saves our data in txt file
    def saveDataToFile(self):
        np.savetxt('./parsedData/rows.txt', self.rows, delimiter=',', fmt="%s")
        np.savetxt('./parsedData/columns.txt', self.columns, delimiter=',', fmt="%s")
        np.savetxt('./parsedData/movieData.txt', self.movieData, delimiter=',')
        return

    # The method reads the data from the txt file
    def readDataFromFile(self):
        self.items = []
        self.user_ratings = {}
        movieData = open("./parsedData/movieData.txt", "r")
        rows = open("./parsedData/rows.txt", "r")
        columns = open("./parsedData/columns.txt", "r")
        self.movieData  = np.loadtxt(movieData, delimiter=',')
        self.rows = np.genfromtxt("./parsedData/rows.txt", delimiter="\n", dtype='str')
        self.columns = np.genfromtxt("./parsedData/columns.txt", delimiter="\n", dtype='str')
        for item in self.columns:
            self.items.append(item)
        for u in range(len(self.rows)):
            ocene_uporabnika = {}
            user = self.rows[u]
            ocene = np.where(self.movieData[u,:] != 0)[0]
            ocenjeni_filmi = self.columns[ocene]
            for kk in range(len(ocene)):
                movie = ocenjeni_filmi[kk]
                rate = self.movieData[u,ocene[kk]]
                ocene_uporabnika[movie] = rate
            self.user_ratings[user] = ocene_uporabnika
        return

    # method prints numpy array
    def printNPdata(self):
        np.set_printoptions(threshold=np.nan)
        np.set_printoptions(linewidth=180)
        print(self.movieData)

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
            self.user_avgs[user] = (sumRatings - global_avg)
        for m in range(len(self.columns)):
            movie = self.columns[m]
            sumRatings = sum(self.movieData[:,m])
            numRatings = np.count_nonzero(self.movieData[:,m])
            self.item_avgs[movie] = (sumRatings - global_avg) 
        return

    # the method prints all users and their average
    def print_users(self):
        for user in self.user_avgs.keys():
            print('name: {0:13}  avg: {1}'.format(user, self.user_avgs[user]))

    # the method prints out all the films and their averages
    def print_items(self):
        for item in self.item_avgs.keys():
            print('name: {0:40}  avg: {1}'.format(item, self.item_avgs[item]))

    # the method returns the naked average of the film
    def predict_simple_average(self, user, item):
        if item in self.item_avgs:
            return self.item_avgs[item];

    # method returns the number of movie views
    def predict_views(self, user, item):
        return np.count_nonzero(self.movieData[:,np.where(self.columns == item)])

    # method returns n movies according to the method method
    def recommendNItems(self, user, method, n):
        temp = [];
        if method ==  self.predict_simple_average:
            temp = [(value, key) for key, value in self.item_avgs.items()]
        elif method == self.predict_views:
            for movie in self.items:
                temp.append((self.predict_views(user, movie), movie));
        sort = sorted(temp, key = lambda tuple: method(user, tuple[1]), reverse = False)
        x = random.randrange(8)
        return sort[x:9]

    # the method returns the most controversial films
    def controversialMovies(self, n):
        cont_list = []
        for movie in self.items:
            ratings = []
            for userData in self.user_ratings.values():
                if movie in userData:
                    ratings.append(userData[movie])
            ratings.sort()
            firstHalf = ratings[:int(len(ratings)/2)]
            secondHalf = ratings[int(len(ratings)/2):]
            cont_ratio = sum(secondHalf)/len(secondHalf) - sum(firstHalf) / len(firstHalf)
            cont_list.append((movie, cont_ratio))
        sort = sorted(cont_list, key = lambda tuple: tuple[1], reverse = True)
        return sort[:n]

    # 2.

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

    # we predict what would, give the user an evaluation of the selected film
    def predict_usersim(self, user, item, threshold = 0.0):
        top_calculation = 0
        normalisation = 0
        mask = np.ma.masked_where(self.movieData == 0, self.movieData)
        row = np.where(self.rows == user)[0][0]
        user_avg = np.mean(mask[row,:], axis=0)
        item_column = np.where(self.columns == item)[0][0]
        for current_row in range(len(self.rows)):
            current_user = self.rows[current_row]
            if current_row != row and self.movieData[current_row, item_column] != 0:
                current_avg = np.mean(mask[current_row,:], axis=0)
                sim = self.user_similarity(user, current_user)
                if(math.isnan(sim) == False):
                    top_calculation += sim * ( self.movieData[current_row, item_column] - current_avg )
                    normalisation += sim
        return user_avg + (top_calculation * 1.0 / normalisation)

    #calculating the similarity of two products by cosine similarity
    def item_similarity(self, item1, item2):
        mask = np.ma.masked_where(self.movieData == 0, self.movieData)
        mean = np.mean(mask, axis=1)
        normalized = np.transpose(np.subtract(np.transpose(mask), mean))
        indeks1 = np.where(self.columns == item1)[0][0]
        indeks2 = np.where(self.columns == item2)[0][0]
        ocene1 = np.where(self.movieData[:,indeks1] != 0)
        ocene2 = np.where(self.movieData[:,indeks2] != 0)
        vse_ocene = np.intersect1d(ocene1, ocene2)
        item1 = normalized[vse_ocene, indeks1]
        item2 = normalized[vse_ocene, indeks2]
        return 1 - cosine(item1, item2)

    # 3.

    # the method returns the list with all the accuracy calculations of the model
    def evaluate(self, data, predictions):
        mae = mean_absolute_error(data, predictions)
        rmse = mean_squared_error(data, predictions)
        recall = recall_score(data, predictions)
        precision = precision_score(data, predictions)
        f1score = f1_score(data, predictions)
        return [mae, rmse, recall, precision, f1score]

    # This method predicts how U would rate product I with the SlopeOne algorithm
    def predict_S1(self, user, item):
        #self.rows = np.array(["A", "B", "C", "D"])
        #self.columns = np.array(["I1", "I2", "I3", "I4"])
        #self.movieData = np.array([[5, 3, 4, 0],[3, 0, 1, 3],[0, 2, 1, 5],[2, 4, 4, 2]])
        #user = "C"
        #item = "I1"
        stolpec = np.where(self.columns == item)[0][0]
        vrstica = np.where(self.rows == user)[0][0]
        stolpec_ocene = np.where(self.movieData[:, stolpec] != 0)
        sestevki = []
        for s in range(len(self.columns)):
            if s != stolpec:
                gledan_stolpec_ocene = np.where(self.movieData[:, s] != 0)
                presek = np.intersect1d(stolpec_ocene, gledan_stolpec_ocene)
                primerjavaSvS = [0, len(presek), s]
                for p in presek:
                    if p != vrstica:
                        primerjavaSvS[0] += (self.movieData[p,stolpec] - self.movieData[p, s])
                sestevki.append(primerjavaSvS)
        zgoraj = 0
        spodaj = 0
        for ses in sestevki:
            zgoraj += (self.movieData[vrstica, ses[2]] + (ses[0] * 1.0 / ses[1])) * ses[1]
            spodaj += ses[1]
        return zgoraj * 1.0 / spodaj

    # 4.

    # generates file basket for binding rules
    def save_basket_data(self, filename):
        basket = []
        for u in range(len(self.rows)):
            user = self.rows[u]
            user_avg = self.user_avgs[user]
            movie_index = np.where(self.movieData[u, :] > user_avg)[0]
            basket.append(self.columns[movie_index].tolist())
        basket = np.array(basket)
        # record in file
        file = open(filename, "w")
        for row in basket:
            vrstica = ""
            for item in row:
                vrstica += item +","
            vrstica = vrstica[:-1] + "\n"
            file.write(vrstica)
        return True

    def save_item_distance_data(self, filename):
        distance = np.array([[None] * len(self.columns)] * len(self.columns))
        for i in range(len(self.columns)):
            for j in range(i, len(self.columns)):
                item1 = self.columns[i]
                item2 = self.columns[j]
                dis12 = 1 - self.item_similarity(str(item1), str(item2))
                distance[i,j] = round(dis12, 4)
        distance = np.transpose(distance)

        # record in file
        file = open(filename, "w")
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