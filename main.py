from GPSPhoto import gpsphoto
import pandas as pd
import numpy as np
import math
from geopy import Nominatim
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import stats
import seaborn
seaborn.set()


def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation

    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)

    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)

    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)

    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def create_gpx(dataframe):
    dataframe['lat1'] = dataframe['lat'].shift(periods=-1)
    dataframe['lon1'] = dataframe['lon'].shift(periods=-1)
    dataframe = dataframe.dropna()
    output_gpx(dataframe , 'predicted.gpx')
    dataframe = dataframe.drop(columns=['lat1','lon1'])

def set_city(df):
    geolocator = Nominatim(user_agent = 'test_1')
    cor = (df['lat'],df['lon'])
    location = geolocator.reverse(cor)
    address = location.raw['address']
    df['address'] = address
    try:
        df['city'] = address['city']
    except KeyError:
        df['city'] = "Null"
    return df

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

def main():
    # Get the data from image file and return a dictionary
    data = {}
    num_photos = 36
    for i in range(1,num_photos):
        data[i] = gpsphoto.getGPSData('provided_image/'+str(i)+'.jpeg')
        data[i]['id'] = i

    df = pd.DataFrame(data)
    df = df.T
    #df.to_csv('latlng.csv', index=False)

    df = df.drop(columns=['Date','Altitude'])

    plt.scatter(df['Latitude'],df['Longitude'])
    plt.savefig('2dwalk_scatter_final.png')

    model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=True),
        LinearRegression(fit_intercept=False)
    )

    model.fit(np.stack([df['Latitude']], axis=1), df['Longitude'])
    x = np.linspace(df['Latitude'].max(), df['Latitude'].min(),100, dtype=np.longfloat)
    y = model.predict(np.stack([x], axis=1))

    plt.scatter(x,y)
    plt.savefig('2dwalk_predicted_scatter_final.png')

    plt.scatter(df['Latitude'],df['Longitude'])
    plt.plot(x , y ,'r-', linewidth=3)
    plt.savefig('2dwalk_predicted_final.png')

    avg_latlon = pd.DataFrame() 
    avg_latlon['lat'] = x
    avg_latlon['lon']  = y

    create_gpx(avg_latlon)

    amenities = pd.read_csv('group_withcity/group1_1.csv')

    avg_latlon = avg_latlon.apply(set_city,axis=1)

    
    joined = avg_latlon.join(amenities.set_index('city'),lsuffix='_caller',on='city')
    joined['distance']  = joined.apply(distance,axis=1)

    all_possible_amenities = pd.DataFrame() 

    all_possible_amenities['lat'] = joined['lat']
    all_possible_amenities['lon'] = joined['lon']
    all_possible_amenities['distance'] = joined['distance']

    all_possible_amenities['amenity'] = joined['amenity']
    all_possible_amenities['city'] = joined['city']

    all_possible_amenities = all_possible_amenities.groupby(['lat','lon']).min()

    suggested = all_possible_amenities[all_possible_amenities['distance']>10]
    suggested = all_possible_amenities[all_possible_amenities['distance']<6000]

    print()
    print("I suggest you could go ")
    print(suggested)
    #joined.to_csv('joined.csv', index=False)

if __name__ == '__main__':
    main()