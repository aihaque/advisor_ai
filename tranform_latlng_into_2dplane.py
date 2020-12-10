import pandas as pd
import numpy as np
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

def find_xy(df):
    R = 6371
    x = R * np.cos(df['Latitude']) * np.cos(df['Longitude'])
    y = R * np.cos(df['Latitude']) * np.sin(df['Longitude'])
    z = R *np.sin(df['Latitude'])
    df['x'] = x
    df['y'] = y
    df['z'] = z
    return df

def rec(df):
    R = 6371

    df['lat'] = np.arcsin(df['z'] / R)
    df['lon'] = np.arctan2(df['y'], df['x'])
    return df

df = pd.read_csv('latlng.csv')
df = df.drop(columns=['Date','Altitude'])
df = df.apply(find_xy,axis=1)
#plt.scatter(df['Latitude'],df['Longitude'])
#plt.savefig('2dwalk_scatter.png')

#filtered = lowess(df['Longitude'], df['Latitude'], frac=0.9)
#plt.scatter(filtered[:, 0], filtered[:, 1])
#plt.show()
#plt.savefig('2dwalk_predicted_notconverted.png')
model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=True),
    LinearRegression(fit_intercept=False)
)
#print(np.stack([df['Latitude']], axis=1))
model.fit(np.stack([df['Latitude']], axis=1), df['Longitude'])
x = np.linspace(df['Latitude'].max(), df['Latitude'].min(),10, dtype=np.longfloat)
y = model.predict(np.stack([x], axis=1))
#plt.scatter(x, model.predict(np.stack([x], axis=1)))
#plt.show()
#print(model.predict(np.stack([df['Latitude']], axis=1)))
#fit = stats.linregress(df['Latitude'], df['Longitude'])
filtered_df = pd.DataFrame() 
filtered_df['lat'] = x
filtered_df['lon']  = y
#plt.plot(filtered_df['lat'], filtered_df['lon'], 'r-', linewidth=3)
#plt.show()
#filtered_df = pd.DataFrame(filtered,columns=['lat','lon'])
#filtered_df['z'] = df[df['id'] =] 
#filtered_df = filtered_df.apply(rec,axis=1)
#filtered_df = filtered_df.drop(columns=['x','y'])
filtered_df['lat1'] = filtered_df['lat'].shift(periods=-1)
filtered_df['lon1'] = filtered_df['lon'].shift(periods=-1)
filtered_df = filtered_df.dropna()

output_gpx(filtered_df , 'predicted.gpx')

filtered_df = filtered_df.drop(columns=['lat1','lon1'])
filtered_df.to_csv('avg_latlng.csv', index=False)
#print( np.linspace(49.2484472222222, 49.28035,100, dtype=np.longfloat) )
