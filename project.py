# %%
# Load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# %%
# Load data
taxi_data = pd.read_csv('taxi_flow_2016-01_2016-02_2016-03.csv', parse_dates=['stime', 'ttime'])

# Extract January only
taxi_data = taxi_data[taxi_data['stime'] < '2016-02-01']

taxi_data.tail()

# %%
#
