import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    summarize_data(car_market)