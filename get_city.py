from geopy import Nominatim
import pandas as pd
import numpy as np
import os

def set_city(df):
    cor = (df['lat'],df['lon'])
    location = geolocator.reverse(cor)
    address = location.raw['address']
    print(cor)
    print(address)
    df['address'] = address
    try:
        df['city'] = address['city']
    except KeyError:
        df['city'] = "Null"
    return df

geolocator = Nominatim(user_agent = 'test_1')
big_df = []
for filename in os.listdir('grouped_new'):
    df = pd.read_csv('grouped_new/'+filename)
    df = df.apply(set_city,axis=1)
    big_df.append(df)

concat_df = pd.concat(big_df)    
concat_df.to_csv('test.csv', index=False)

#avg_df = pd.read_csv('avg_latlon.csv')
#avg_df = avg_df .apply(set_city,axis=1)
#avg_df.to_csv('avg_latlon.csv', index=False)

#cor = (49.1340348, -122.3040006)
#location = geolocator.reverse(cor)
#address = location.raw['address']
#print(address)
