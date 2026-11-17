from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging


from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder,RobustScaler,StandardScaler
import os
import json
from datetime  import datetime
from TM1py.Services import TM1Service
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from .utils.all_utils import *
import argparse

#SARIMA
from time import process_time
import seaborn as sns
from dateutil.relativedelta import relativedelta
import pmdarima as pmd
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")
load_env_file('.env')
user = os.getenv('USER')
password = os.getenv('PASSWORD')
namespace = os.getenv('NAMESPACE')
address = os.getenv('ADDRESS')
gateway = os.getenv('GATEWAY')
port = os.getenv('PORT')
ssl = os.getenv('SSL')
cube_name=os.getenv('CUBE_NAME')
view_name=os.getenv('VIEW_NAME')
base_dir_cdsw=os.getenv('BASE_DIR_CDSW_DATAMART')
config_columns=load_config_json("job-scripts/list_columns.json")

default_columns_simpanan=config_columns['default_columns_simpanan']
default_columns_pinjaman=config_columns['default_columns_pinjaman']
measure_dict = {
    "simpanan": {
        "ro": config_columns['simpanan_measure_ro'],
        "nas": config_columns['simpanan_measure_nas']
    },
    "pinjaman": {
        "ro": config_columns['pinjaman_measure_ro'],
        "nas": config_columns['pinjaman_measure_nas']
    }
}