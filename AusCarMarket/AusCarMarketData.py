import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

path = "../Datasets/AusCarMarket/cars_info.csv"
car_market = pd.read_csv(path)

conversion_rate = 0.67
car_market['Price_USD'] = car_market['Price'] * conversion_rate

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
    sns.histplot(df['Price_USD'], kde=False, bins=100)
    plt.title("Price Distribution")

    # In Python, _ is commonly used as a placeholder variable name to indicate that
    # the value it holds is not going to be used.
    formatter = FuncFormatter(lambda x, _: f'${x / 1e6:.1f}M' if x >= 1e6 else f'${x / 1e3:.0f}K')
    # gca = get current axis
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.show()


def brand_polarity(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Brand', order=df['Brand'].value_counts().index)
    plt.title('Brand Popularity')
    plt.xticks(rotation=90)
    plt.show()


def fuel_type_distribution(df):
    plt.figure(figsize=(15, 6))
    # here x is the column name that will be plotted in the x axis
    sns.countplot(data=df, x='Fuel')
    plt.title('Fuel Type Distribution')
    plt.xticks(rotation=90)
    plt.show()


def price_year_comparison(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Year', y='Price_USD')
    plt.title('Price vs. Year')
    plt.show()


def price_km_comparison(df):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='Kilometers', y='Price_USD')
    plt.title('Price vs. Kilometers')
    plt.show()


def price_brand_comparison(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Brand', y='Price_USD')
    plt.title('Price by Brand')
    plt.xticks(rotation=90)
    plt.show()


if __name__ == "__main__":
    price_brand_comparison(car_market)