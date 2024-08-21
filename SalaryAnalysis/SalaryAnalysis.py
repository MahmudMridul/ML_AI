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

