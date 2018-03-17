import Recommender

r = Recommender.Recommender()
#r.parse_database("./data/moviebase2014.txt")
#r.toNumpy()
#r.saveDataToFile()

r.readDataFromFile()
r.compute_averages()

r.printNPdata()
print("")

print("Items: ", r.items)
print("")
print(r.user_ratings)

print("")
print("Users:-")
r.print_users()
print("")

print("Items:-")
r.print_items()
print("")

user=input("Enter User: ")
movie=input("Enter Movie: ")
print("Average Rating for "+movie+": ")
print(r.predict_simple_average(user, movie))
print("")

print("RecommendNItems-simple_average:")
print(r.recommendNItems(user, r.predict_simple_average, 5))
print("")

print("RecommendNItems-predict_views:")
print(r.recommendNItems(user, r.predict_views, 5))
print("")

print("Top 5 Controversial Movies:-")
print(r.controversialMovies(5))
print("")
print("User Similarity:-")
user1=input("Enter User1: ")
user2=input("Enter User2: ")
print("")
print(r.user_similarity(user1, user2))
print("")

print("Predict user rating:-")
user=input("Enter user: ")
movie=input("Enter movie: ")
print("")
print(+r.predict_usersim(user,movie))
print("")

print("User cosine similarity:-")
movie1=input("Movie1: ")
movie2=input("Movie2: ")
print("")
print(r.item_similarity(movie1,movie2))
print("")

print("Slope One")
print(r.predict_S1(user,movie))
print("")

print("Association rules")
print(r.save_basket_data("./output/movies.basket"))
print("")

print("Distance matrix")
print(r.save_item_distance_data("./output/distanceMatrix.dst"))
print("")
