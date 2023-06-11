from laptopPrice import data_processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import operator

"""
I don't want to have to redo all of the work I did in the regression version, so I'll import the data processing
to speed up the KNN implementation.
"""


def compute_distance(a, b):
    a_ram = a[0]
    b_ram = b[0]
    ram_distance = abs(a_ram - b_ram)
    a_graphic = a[1]
    b_graphic = b[1]
    graphic_distance = abs(a_graphic - b_graphic)
    a_rating = a[2]
    b_rating = b[2]
    rating_distance = abs(a_rating - b_rating)
    normalized_distance = (ram_distance + graphic_distance + rating_distance) / \
                          (a_ram+b_ram+a_graphic+b_graphic+a_rating+b_rating)
    return normalized_distance


def get_neighbors(df_id, K, train_test):
    distances = []
    for j in range(len(X_train)):
        if train_test == "train":
            if j != df_id:
                dist = compute_distance(X_train.iloc[df_id], X_train.iloc[j])
                distances.append((j, dist))
        if train_test == "test":
            dist = compute_distance(X_test.iloc[df_id], X_train.iloc[j])
            distances.append((j, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for k in range(K):
        neighbors.append(distances[k][0])
    return neighbors


# import the dataset
df = pd.read_csv("laptopPrice.csv")
# process the dataset
df = data_processing(df)
# train test split the dataset
X = df[["ram_gb", "graphic_card_gb", "rating"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("X[0] is: ", X.iloc[0])
X_update = X_train.append(y_train)
print(X_train, y_train)
# I want to test the compute distance function to make sure it works,
# so I will pick three points, two that are similar and one that is different and compare them

for i in range(len(df)):
    ram = df.iloc[i, 4]
    graphic = df.iloc[i, 10]
    rating = df.iloc[i, 16]
    print(ram, graphic, rating)
    if i > 5:
        break

# rows one and two are similar with values of (4, 0, 2) and (4, 0 , 3)
# row 4 is different with values (8, 2, 3)
# We would expect one and two to have a low distance while one and three/two and three will
# have a larger distance


one_two_distance = compute_distance(X.iloc[0], X.iloc[1])
one_three_distance = compute_distance(X.iloc[0], X.iloc[3])
two_three_distance = compute_distance(X.iloc[1], X.iloc[3])
print("one_two_distance is: ", one_two_distance)
print("one_three_distance is: ", one_three_distance)
print("two_three_distance is: ", two_three_distance)

# it works, the smaller the value the more similar the two values are.

K = 10
avg_price = 0
neighbors = get_neighbors(1, K, "train")
for neighbor in neighbors:
    avg_price += y_train.iloc[neighbor]
avg_price /= float(K)
print("avg_price is: ", avg_price)
# returned a value of 29422.7
print("real price is: ", y_train.iloc[1])
# returned a value of 38999

# The prediction is close but not really close enough, if the K value is better
# the prediction could be closer to the actual so multiple K values need to be checked

for i in range(20):
    neighbors = get_neighbors(1, i+1, "train")
    for neighbor in neighbors:
        avg_price += y.iloc[neighbor]
    avg_price /= float(i+1)
    print("avg_price is: ", avg_price, "for run: ", i+1)
    print("real price is: ", y[1])
    diff = abs(avg_price - y[1])
    print("diff is: ", diff)
    print(" ")

# A K value of 5 seems to be the closest for predicting price
y_pred = []
y_percent_error = []
for i in range(len(X_test)):
    avg_price = 0
    neighbors = get_neighbors(i, 10, "test")
    for neighbor in neighbors:
        avg_price += y_train.iloc[neighbor]
    avg_price /= 10
    y_pred.append(avg_price)
    percent_error = abs((avg_price-y_test.iloc[i])/y_test.iloc[i])*100
    y_percent_error.append(percent_error)
    print("avg_price: ", avg_price, "real price: ", y_test.iloc[i], "percent error is: ", y_percent_error[i])

print("average percent error is: ", np.mean(y_percent_error))

# A percent error of roughly 30-40% isn't great but it could also be a lot worse.
# I think that if I were to keep moving forward on this project I would have to implement
# a neural network that could be trained to catch the nuances of the dataset better than
# KNN was able to. Also my implementation of the KNN algorithm could probably use refining.
# However, I think that this is fine for now and I would like to develop other projects and
# might come back to develop the neural network for this in the future.

