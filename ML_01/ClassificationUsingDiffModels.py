import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf

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
# lr_model = LogisticRegression()
# lr_model.fit(x_train, y_train)
# y_pred = lr_model.predict(x_test)
# print(classification_report(y_test, y_pred))

# Support Vector Machine
# sv_model = SVC()
# sv_model.fit(x_train, y_train)
# y_pred = sv_model.predict((x_test))
# print(classification_report(y_test, y_pred))

# Neural Network

def plot_history(history):
    fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plot.show()


def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                     metrics=['accuracy'])
    history = nn_model.fit(
        X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
    )

    return nn_model, history


least_val_loss = float('inf')
least_loss_model = None
epochs = 100
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                model, history = train_model(x_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                plot_history(history)
                val_loss = model.evaluate(x_valid, y_valid)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model


y_pred = least_loss_model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)

