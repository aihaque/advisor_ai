from GPSPhoto import gpsphoto
import pandas as pd
import numpy as np
import math
import mplleaflet
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

def user_input(data):
    df = pd.read_csv('categories.csv')
    print('Please choose what type of amenity you are interested in: ')
    print('0. In general')
    print('1. Financial')
    print('2. Vehicles')
    print('3. Health')
    print('4. Recreation')
    print('5. Food')
    print('6. Work')
    print('7. Pray')
    print('8. Trash')
    print('9. Repair')
    print('10. Clock')
    print('11. Landmark')
    print('12. transport')

    value = input("")
    value = int(value)
    category = "N/A"
    if(value == 1):
        category = "money"
    elif(value == 2):
        category = "vehicles"
    elif(value == 3):
        category = "health"
    elif(value == 4):
        category = "recreation"
    elif(value == 5):
        category = "food"
    elif(value == 6):
        category = "work"
    elif(value == 7):
        category = "pray"
    elif(value == 8):
        category = "trash"
    elif(value == 9):
        category = "repair"
    elif(value == 10):
        category = "clock"
    elif(value == 11):
        category = "landmarks"
    elif(value == 12):
        category = "transports"
    if(category!="N/A"):
        df_category = df[category].dropna()
        print(df_category)
        data = data[data['amenity'].isin(df_category)]
    return data

def city_input():
    print('Please choose for which city you are interested in: ')
    print('1. Burnaby')
    print('2. Surrey')
    print('3. Coquitlam')
    print('4. Richmond')
    print('5. Vancouver')
    print('6. North Vancouver')
    print('7. West Vancouver')
    print('8. Delta')

    value = input("")
    value = int(value)
    category = "N/A"
    if(value == 1):
        category = "Burnaby"
    elif(value == 2):
        category = "Surrey"
    elif(value == 3):
        category = "Coquitlam"
    elif(value == 4):
        category = "Richmond"
    elif(value == 5):
        category = "Vancouver"
    elif(value == 6):
        category = "North Vancouver"
    elif(value == 7):
        category = "West Vancouver"
    elif(value == 8):
        category = "Delta"
    return category
    
def executeAI(avg_latlon,amenities):

    joined = avg_latlon.join(amenities.set_index('city'),lsuffix='_caller',on='city')
    joined['distance']  = joined.apply(distance,axis=1)

    all_possible_amenities = pd.DataFrame() 

    all_possible_amenities['lat'] = joined['lat']
    all_possible_amenities['lon'] = joined['lon']
    all_possible_amenities['distance'] = joined['distance']

    all_possible_amenities['amenity'] = joined['amenity']
    all_possible_amenities['city'] = joined['city']
    all_possible_amenities['address'] = joined['address']

    all_possible_amenities = all_possible_amenities.groupby(['lat','lon']).min()

    suggested = all_possible_amenities[all_possible_amenities['distance']>10]
    suggested = all_possible_amenities[all_possible_amenities['distance']<2000]

    if(suggested.empty):
        print("Sorry nothing to suggest")
    else:
        print()
        print("I suggest you could go ")
        
        suggested = suggested.reset_index()
        p = suggested.drop(columns=['lat','lon'])
        print(p)

        plt.figure(figsize=(8,6))
        fig = plt.figure()
        #x = suggested[suggested['amenity'] == 'atm']
        #x = x.reset_index()
        #plt.scatter(x['lon'],x['lat'])

        plt.scatter(suggested['lon'],suggested['lat'])

        filtered = lowess(suggested['lat'],suggested['lon'], frac=0.2)
        plt.plot(filtered[:, 0], filtered[:, 1],'r-', linewidth=5)

        #plt.plot(x,y, 'r-', linewidth=3)
        mplleaflet.show(fig=fig)


def execute_nightclub_AI(nightclub,amenities):

    joined = nightclub.join(amenities.set_index('city'),lsuffix='_caller',on='city')
    joined['distance']  = joined.apply(distance,axis=1)
    joined = joined[joined['distance']<1000]

    joined['count'] = joined.groupby(['lat_caller','lon_caller'])['tags'].transform('count')
    joined = joined[joined['count']>10]

    suggested = pd.DataFrame()
    suggested = joined
    suggested = suggested.drop_duplicates(subset=['lat_caller','lon_caller'])

    print('Suggested NightClubs info')
    print(suggested['address_caller'])

    long_info = pd.DataFrame()
    long_info['Suggested NightClubs info'] = joined['address_caller']
    long_info['Aminities_Within_1Km'] = joined['amenity']
    print(long_info)

    plt.figure(figsize=(8,6))
    fig = plt.figure()
    plt.scatter(joined['lon_caller'],joined['lat_caller'],30)
    plt.scatter(joined['lon'],joined['lat'],10)
    mplleaflet.show(fig=fig)



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
    x = np.linspace(df['Latitude'].max(), df['Latitude'].min(),10, dtype=np.longfloat)
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

    amenities = pd.read_csv('cities/Burnaby.csv')
    amenities = user_input(amenities)

    #avg_latlon = avg_latlon.apply(set_city,axis=1)
    avg_latlon = pd.read_csv('avg_latlon.csv')

    if(amenities.empty or avg_latlon.empty):
        print("Sorry not any popular amenities nearby")
    else:
        executeAI(avg_latlon,amenities)

    print("I might help you with one more adviced if you will be planning to go a nightclub")
    print("0 Not interested")
    print("1 Interested")
    value = input("")
    value = int(value)
    if(value==1):
        city = city_input()
        amenities = pd.read_csv('cities/'+ city + '.csv')
        nightclub = amenities[amenities['amenity'] == 'nightclub']

        category = ['casino','park','bar','stripclub','pub','hospital','atm']
        amenities = amenities[amenities['amenity'].isin(category)]
        if(amenities.empty or nightclub.empty):
            print("Sorry no popular nightclub found in your city")
        else:
            execute_nightclub_AI(nightclub,amenities)
        

if __name__ == '__main__':
    main()