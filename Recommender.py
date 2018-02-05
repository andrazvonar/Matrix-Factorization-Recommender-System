import numpy as np
import pickle


def read_file(filename):
    # loads file without header
    parsed_file = np.loadtxt(filename)
    return parsed_file


def write_file(data, filename):
    file = open(filename, 'w', encoding="utf-8")
    for line in data:
        file.write(str(line) + "\n")

    file.close()


class Recommender:
    def __init__(self, training, test, data, K, alpha, eta):
        """
        Prepares environment for model computation.
        :param training: training dataset
        :param test: testing dataset
        :param data: whole dataset
        :param K: factorization factor
        :param alpha: learning factor
        :param eta: regularization factor
        """
        self.training = training
        self.test = test
        self.K = K
        self.alpha = alpha
        self.eta = eta
        self.data = data

        # Build a list of all distinct users and artists
        self.users = np.unique([x for x, y, z in self.training])
        self.artists = np.unique([y for x, y, z in self.training])

        # Import user average ratings to speed up recommendation
        with open('Data/user_averages.pkl', 'rb') as f:
            self.user_averages = pickle.load(f)

        # Import friendships
        friends = read_file("/Users/andraz/Documents/šola/UOZP/naloga 5/Data/extra/user_friends.dat")
        self.friendships = {}

        for friendship in friends:
            if friendship[0] in self.friendships:
                self.friendships[friendship[0]].append(friendship[1])
            else:
                self.friendships[friendship[0]] = [friendship[1]]

        # Reorganize ratings for easier access
        self.ratings = {}
        for profile in self.data:
            if profile[0] in self.ratings:
                self.ratings[profile[0]][profile[1]] = profile[2]
            else:
                self.ratings[profile[0]] = {profile[1]: profile[2]}

        # Initialize output matrices
        self.P = np.random.uniform(low=0, high=np.sqrt(10/self.K), size=(len(self.users), self.K))
        self.Q = np.random.uniform(low=0, high=np.sqrt(10/self.K), size=(self.K, len(self.artists)))

        # Account for bias
        self.P[:, 0] = np.ones(len(self.users))
        self.Q[1, :] = np.ones(len(self.artists))

        # Run ISMF algorithm until convergence
        self.rmses = []

        print("Fitting model ...")
        while self.converging():
            self.ISMF()
            self.rmses.append(self.RMSE())
            print(str(self.rmses[-1]))

    def user_average(self, u):
        return np.mean([self.recommend_internal(u, a) for a in self.artists])

    def converging(self):
        """
        Tests if RMSE is converging.
        :return: Returns False if last RMSE is larger that last 2. True otherwise.
        """
        if len(self.rmses) >= 3 and self.rmses[-2] < self.rmses[-1]:
            return False
        else:
            return True

    def e(self, p_u, q_i, r_ui):
        return 0.5 * (r_ui - np.dot(p_u, q_i))

    def RMSE(self):
        real = []
        pred = []
        for x, y, z in self.test:
            if len(np.where(self.users == x)[0]) > 0 and len(np.where(self.artists == y)[0]) > 0:
                real.append(z)
                pred.append(self.recommend_internal(x, y))

        real = np.array(real)
        pred = np.array(pred)

        return np.sqrt(np.mean((real - pred) ** 2))

    def ISMF(self):
        for u, i, r_ui in self.training:
            # Find matrix indices of current user and artist
            u = np.where(self.users == u)[0][0]
            i = np.where(self.artists == i)[0][0]

            # Compute error
            error = self.e(self.P[u, :], self.Q[:, i], r_ui)

            # Compute gradient and improve model
            self.P[u, :] = self.P[u, :] + self.alpha * (error * self.Q[:, i] - self.eta * self.P[u, :])
            self.Q[:, i] = self.Q[:, i] + self.alpha * (error * self.P[u, :] - self.eta * self.Q[:, i])

    def recommend_internal(self, user, artist):
        """
        Predicts artist rating, used for internal RMSE computation.
        :param user: user name
        :param artist: artist name
        :return: artist rating prediction
        """
        u = np.where(self.users == user)[0]
        i = np.where(self.artists == artist)[0]

        if len(u) > 0 and len(i) > 0:
            u = u[0]
            i = i[0]

            return np.dot(self.P[u, :], self.Q[:, i])

    def recommend(self, user, artist):
        """
        Predicts artist rating. Uses friendship data to improve accuracy.
        :param user: user name
        :param artist: artist name
        :return: artist rating prediction
        """
        u = np.where(self.users == user)[0]
        i = np.where(self.artists == artist)[0]

        # Ratio controls significance of friendship data in prediction
        ratio = 0.20
        if len(u) > 0 and len(i) > 0:
            u = u[0]
            i = i[0]
            friend_rating = self.get_avg_friend_rating(user, artist)

            if friend_rating != 0:
                return ((1-ratio)*np.dot(self.P[u, :], self.Q[:, i])) + (ratio*friend_rating)
            else:
                return np.dot(self.P[u, :], self.Q[:, i])

        elif len(u) > 0:
            friend_rating = self.get_avg_friend_rating(user, artist)
            if friend_rating != 0:
                return (0.5*self.user_averages[user]) + (0.5*friend_rating)
            else:
                return self.user_averages[user]
        else:
            return 1.82106569656

    def get_avg_friend_rating(self, user, artist):
        """
        Computes an average rating user's friends gave the artist
        :param user: User name
        :param artist: Artist name
        :return: Returns average rating or 0 if no data is applicable
        """
        friends = self.friendships[user]
        ratings = []
        for friend in friends:
            if artist in self.ratings[friend]:
                ratings.append(self.ratings[friend][artist])

        if len(ratings) > 0:
            return np.mean(ratings)
        else:
            return 0

    def get_all_recommendations(self, user):
        return [[user, a, self.recommend(user, a)] for a in self.artists]


data = read_file("/Users/andraz/Documents/šola/UOZP/naloga 5/Data/user_artists_training.dat")

# (1) Tekmovanje
if True:
    split = int(len(data) * 0.95)
    training = data[:split]
    test = data[split:]
    rec = Recommender(training, test, data, K=2000, alpha=0.01, eta=0.001)

    test_data = read_file("/Users/andraz/Documents/šola/UOZP/naloga 5/Data/user_artists_test.dat")
    result = []
    print("Computing predictions")
    for user, artist in test_data:
        result.append(rec.recommend(user, artist))

    write_file(result, "/Users/andraz/Documents/šola/UOZP/naloga 5/Data/result.txt")

# (2) Test na podatkih
if False:
    split = int(len(data) * 0.70)
    training = data[:split]

    # Test data
    test = data[split:]

    # Training data
    split = int(len(training) * 0.90)
    learning = training[:split]
    convergence = training[split:]

    rec = Recommender(learning, convergence, data, K=2000, alpha=0.01, eta=0.001)
    print(np.sqrt(np.mean([(rating - rec.recommend(user, artist))**2 for user, artist, rating in test])))

# (3) Moji podatki
if False:
    split = int(len(data) * 0.90)
    training = data[:split]
    convergence = data[split:]

    mydata = np.loadtxt("/Users/andraz/Documents/šola/UOZP/naloga 5/Data/my_training_data.txt")
    training = np.append(training, mydata, axis=0)

    rec = Recommender(training, convergence, training, K=500, alpha=0.01, eta=0.001)
    result = sorted(rec.get_all_recommendations(5000), key=lambda x: x[2], reverse=True)
    result = result[:10]

    file = open("/Users/andraz/Documents/šola/UOZP/naloga 5/Data/artists.dat", encoding="utf-8")
    artist_names = {}
    for line in file:
        line = line.split('\t')
        artist_names[float(line[0])] = line[1]

    result = [(artist_names[artist_id], rating) for _, artist_id, rating in result]

    write_file(
        result,
        "/Users/andraz/Documents/šola/UOZP/naloga 5/Data/my_result.txt"
    )
    print("Done.")
