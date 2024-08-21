import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import iplot, plot
from plotly.subplots import make_subplots
from matplotlib.colors import ListedColormap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings

warnings.filterwarnings('ignore')

colors = ["#b1e7cd", "#854442", "#000000", "#fff4e6", "#3c2f2f",
          "#be9b7b", "#512E5F", "#45B39D", "#AAB7B8 ", "#20B2AA",
          "#FF69B4", "#00CED1", "#FF7F50", "#7FFF00", "#DA70D6"]

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

filePath = "../Datasets/DSSalary/ds_salaries.csv"
df = pd.read_csv(filePath)


# summarize the dataset
def summarize_dataset(data_frame):
    print(f"Rows: {data_frame.shape[0]}\nColumns: {data_frame.shape[1]}")
    print(data_frame.info())
    summary = pd.DataFrame({
        'count': data_frame.shape[0],
        'nulls': data_frame.isnull().sum(),
        'null(%)': data_frame.isnull().mean() * 100,
        'cardinality': data_frame.nunique()
    })
    print(summary)
    print(data_frame.describe(include=np.number))
    print(data_frame.describe(include=object))


def sum_of_salary_per_year_line_graph(data_frame):
    sum_of_salary_per_year = data_frame.groupby('work_year')['salary'].sum()

    fig = px.line(
        sum_of_salary_per_year,
        x=sum_of_salary_per_year.index,
        y=sum_of_salary_per_year.values,
        title='Sum of Salary per Year',
        labels={'x': 'Work Year', 'y': 'Sum of Salary'},
        markers=True,
        color_discrete_sequence=[colors[4]],
        template='plotly_dark'
    )
    fig.show()


def first_ten_job_count(data_frame):
    first_ten_jobs = data_frame['job_title'].value_counts()[:10]
    fig = px.scatter(
        x=first_ten_jobs.index,
        y=first_ten_jobs.values,
        title='First Ten Job Count',
        color_discrete_sequence=[colors[3]],
        size=first_ten_jobs,
        labels={'x': 'Job Title', 'y': 'Count of Job Title'},
        template='plotly_dark'
    )
    fig.show()


functions = [
    summarize_dataset,
    sum_of_salary_per_year_line_graph,
    first_ten_job_count
]

if __name__ == "__main__":
    prompt = '''
    summarize_dataset - 1
    sum_of_salary_per_year_line_graph - 2
    first_ten_job_count - 3
    exit - 0
    
    '''
    run_program = True
    last_option = 2
    while run_program:
        input_string = input(prompt)
        number = int(input_string)

        if number == 0:
            run_program = False
        elif 1 <= number <= len(functions):
            functions[number - 1](df)
        else:
            print("Invalid Input")

