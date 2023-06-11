import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df_1 = pd.read_csv("laptopPrice.csv")


def data_processing(df):
    for i in range(len(df)):
        gb_array = df.iloc[i, 4].split(" ")
        graphic_array = df.iloc[i, 10].split(" ")
        rating_array = df.iloc[i, 16].split(" ")
        gb_num = int(gb_array[0])
        graphic_num = int(graphic_array[0])
        rating_num = int(rating_array[0])
        df.iloc[i, 4] = gb_num
        df.iloc[i, 10] = graphic_num
        df.iloc[i, 16] = rating_num
    return df


# one_hot_encoded_df = pd.get_dummies(df, columns=["weight"])
"""
plt.scatter(df_1[["ram_gb"]], df_1[["Price"]])
plt.xlabel("ram_gb")
plt.ylabel("Price")
plt.show()

plt.scatter(df_1[["graphic_card_gb"]], df_1[["Price"]])
plt.xlabel("graphic_card_gb")
plt.ylabel("Price")
plt.show()

plt.scatter(df_1[["rating"]], df_1[["Price"]])
plt.xlabel("rating")
plt.ylabel("Price")
plt.show()

for col in df_1.columns:
    print(col)
lin_reg = LinearRegression()
print(df_1.head())
X = df_1[["ram_gb", "graphic_card_gb", "rating"]]
y = df_1["Price"]

# X[["ram_gb", "rating", "graphic_card_gb"]] = scale.fit_transform(X[["ram_gb", "rating", "graphic_card_gb"]].values)
X_train, X_test, y_train, y_test = train_test_split(X, y)

lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
accuracy = r2_score(y_test, y_pred)*100
print("The accuracy of the model is %.2f" % accuracy)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
"""
""" 
I attempted to implement a linear regression on this dataset, however the score I got was extremely low.
After visualizing the data it clearly is not linear. In hindsight I should have visualized the data first
to try and spot any trends before committing the time to building the entire regression implementation.
This explains why the prediction accuracy is so low. Moving forward I should try implementing a KNN which
might be able to capitalize on the natural clustering of the data. If the number of neighbors is set correctly 
KNN could potentially offer good results with this dataset.
"""
