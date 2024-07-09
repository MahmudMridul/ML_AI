import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

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

print((train["class"] == 1).sum())
print((train["class"] == 0).sum())


def scale_dataset(dataframe, oversampling):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    if oversampling:
        sampler = RandomOverSampler()
        x, y = sampler.fit(x, y)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y_reshaped = np.reshape(y, (-1, 1))

    data = np.hstack((x, y_reshaped))
    return data, x, y


train, x_train, y_train = scale_dataset(train, oversampling=True)
valid, x_valid, y_valid = scale_dataset(valid, oversampling=False)
test, x_test, y_test = scale_dataset(test, oversampling=False)
