import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
dataPath = "../Datasets/Magic/magic04.data"
df = pd.read_csv(dataPath, names=cols)

# get distinct values a class
# print(df["class"].unique())

# df["class"] == "g" this part returns a series of True and False
# astype converts them to 0 or 1
# returned series is saved as df["class"]
df["class"] = (df["class"] == "g").astype(int)

# label = "fWidth"
# plot.hist(df[df["class"] == 1][label], color="blue", label="gamma", alpha=0.7, density=True)
# plot.hist(df[df["class"] == 0][label], color="red", label="hadron", alpha=0.7, density=True)
# plot.title(label)
# plot.ylabel("Probability")
# plot.xlabel(label)
# plot.legend()
# plot.show()

train, valid, test = np.split(df.sample(frac=1), [int(0.6 * len(df)), int(0.8 * len(df))])

# print((train["class"] == 1).sum())
# print((train["class"] == 0).sum())


def scale_dataset(dataframe, oversampling):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversampling:
        sampler = RandomOverSampler()
        x, y = sampler.fit_resample(x, y)

    y_reshaped = np.reshape(y, (-1, 1))

    data = np.hstack((x, y_reshaped))
    return data, x, y


train, x_train, y_train = scale_dataset(train, oversampling=True)
valid, x_valid, y_valid = scale_dataset(valid, oversampling=False)
test, x_test, y_test = scale_dataset(test, oversampling=False)

# KNN

# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(x_train, y_train)
#
# y_pred = knn_model.predict(x_test)
#
# print(classification_report(y_test, y_pred))

# Naive Bayes
# nb_model = GaussianNB()
# nb_model.fit(x_train, y_train)
#
# y_pred = nb_model.predict(x_test)
# print(classification_report(y_test, y_pred))

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print(classification_report(y_test, y_pred))