import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#x train and x test: from x_0 to x_14907 -> float
#x train and x test: from x_14908 to x_14957 -> object/string
x_train = pd.read_csv('./A5_2025_train.csv')
x_test = pd.read_csv('./A5_2025_test.csv')


cat_cols = x_train.columns[14908:]

x_train[cat_cols].to_csv("A5_train_category.csv", index=False)
x_test[cat_cols].to_csv("A5_test_category.csv", index=False)



#there is 5 unique suffixe per columns each with the same prefix

#suffix don't increase with the index