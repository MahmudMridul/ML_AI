import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../Datasets/AusCarMarket/cars_info.csv"
car_market = pd.read_csv(path)

def summarize_data(df):
    print(df.info())
    print('***************************************************************************************************')
    print(df.isnull().sum())
    print('***************************************************************************************************')
    print(df.head(20))
    print('***************************************************************************************************')
    print(df.describe())


# The number of bins you choose affects the appearance of the histogram
# More bins: Leads to a more detailed histogram that might show more variability in the data but can also look noisier.
# Fewer bins: Produces a smoother histogram that might hide details but makes the overall pattern easier to see.
def price_distribution(df):
    plt.figure(figsize=(10,5))
    sns.histplot(df['Price'], kde=False, bins=100)
    plt.title("Price Distribution")

    # In Python, _ is commonly used as a placeholder variable name to indicate that
    # the value it holds is not going to be used.
    formatter = FuncFormatter(lambda x, _: f'${x / 1e6:.1f}M' if x >= 1e6 else f'${x / 1e3:.0f}K')
    # gca = get current axis
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.show()


if __name__ == "__main__":
    price_distribution(car_market)