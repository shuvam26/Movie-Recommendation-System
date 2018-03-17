class SlopeOne:

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

    def suggestedRating(self, users, items, averages, targetUserId, targetItemId):
        runningRatingCount = 4.9406564584124654e-324
        weightedRatingTotal = 0.0
        for i in users[targetUserId]:
            ratingCount = self.usersWhoRatedBoth(users, i, targetItemId)
            weightedRatingTotal += (users[targetUserId][i] + averages[(targetItemId, i)]) * ratingCount
            runningRatingCount += ratingCount
        return weightedRatingTotal / runningRatingCount

    def usersWhoRatedBoth(self, users, itemId1, itemId2):
        count = 0
        for userId in users:
            if itemId1 in users[userId] and itemId2 in users[userId]:
                count += 1
        return count

"""
users = {1: {"A":5, "B":3, "C":2},
         2: {"A":3, "B":4},
         3: {"B":2, "C":5}}

items = {"A": {1:5, 2:3},
         "B": {1:3, 2:4, 3:2},
         "C": {1:2, 3:5}}

s = SlopeOne()

averages = {}
averages = s.buildAverageDiffs(items, users)
print("Averages: ", averages)


print({'ItemCount': len(items), 'UserCount': len(users), 'AverageDiffsCount': len(averages)} )
print("Guess that user A will rate item 3= " + str(s.suggestedRating(users,items, averages, 3, 'A')))
"""