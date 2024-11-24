import json
import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels as sm
from scripts.data_cleaner import clean_data, save_cleaned_data, clean_data_main

print(tf.__version__)

if __name__ == '__main__':
    #clean_data_main()
    data = pd.read_csv('data/cleaned_data.csv')
    #print(data.head())
    