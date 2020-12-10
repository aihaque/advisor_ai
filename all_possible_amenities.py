import pandas as pd
import numpy as np
import math

def distance(dataframe):   # https://www.movable-type.co.uk/scripts/latlong.html
    lat1 = dataframe['lat_caller']
    lon1 = dataframe['lon_caller']
    lat2 = dataframe['lat']
    lon2 = dataframe['lon']

    R = 6371 * 1000
    theta = lat1 * (math.pi/180)
    lamda = lat2 * (math.pi/180)
    del_theta = (lat2-lat1) * (math.pi/180)
    del_lamda = (lon2-lon1) * (math.pi/180)

    a = math.sin(del_theta/2) * math.sin(del_theta/2) + math.cos(theta) * math.cos(lamda) * math.sin(del_lamda / 2) * math.sin(del_lamda / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d

avg_latlon = pd.read_csv('avg_latlon.csv')

amenities = pd.read_csv('group_withcity/social_centre.csv')
joined = avg_latlon.join(amenities.set_index('city'),lsuffix='_caller',on='city')
joined['distance']  = joined.apply(distance,axis=1)

all_possible_amenities = pd.DataFrame() 

all_possible_amenities['lat'] = joined['lat']
all_possible_amenities['lon'] = joined['lon']
all_possible_amenities['distance'] = joined['distance']

all_possible_amenities['amenity'] = joined['amenity']
all_possible_amenities['city'] = joined['city']

print(all_possible_amenities)
print(all_possible_amenities.groupby(['lat','lon']).min())
#joined.to_csv('joined.csv', index=False)