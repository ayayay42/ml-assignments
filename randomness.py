import pandas as pd

## df = data frame with 100 observations

data = {}

for i in range (10):
    data[i] = df.sample(n=25, random_state=i)
