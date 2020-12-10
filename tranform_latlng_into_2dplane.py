import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

def find_xy(df):
    R = 6371
    x = R * np.cos(df['Latitude']) * np.cos(df['Longitude'])
    y = R * np.cos(df['Latitude']) * np.sin(df['Longitude'])
    df['x'] = x
    df['y'] = y
    return df
df = pd.read_csv('latlng.csv')
df = df.drop(columns=['Date','Altitude'])
df = df.apply(find_xy,axis=1)
plt.scatter(df['x'],df['y'])
plt.savefig('2dwalk_scatter.png')