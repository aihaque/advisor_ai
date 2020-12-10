from geopy import Nominatim
import pandas as pd
import numpy as np

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
df = pd.read_csv('new/social_centre')
df = df.apply(set_city,axis=1)
df.to_csv('group_withcity/social_centre.csv', index=False)

#avg_df = pd.read_csv('avg_latlon.csv')
#avg_df = avg_df .apply(set_city,axis=1)
#avg_df.to_csv('avg_latlon.csv', index=False)

#cor = (49.1340348, -122.3040006)
#location = geolocator.reverse(cor)
#address = location.raw['address']
#rint(address)
