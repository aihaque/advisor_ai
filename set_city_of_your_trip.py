from geopy import Nominatim
from GPSPhoto import gpsphoto
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

# Get the data from image file and return a dictionary
data = {}
value = input("Please provide the number of trip photos you uploaded\n")
num_photos = int(value)
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

avg_latlon = avg_latlon.apply(set_city,axis=1)
avg_latlon = avg_latlon.drop(columns=['lat1','lon1'])
avg_latlon.to_csv('avg_latlon.csv', index=False)