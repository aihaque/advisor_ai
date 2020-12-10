from GPSPhoto import gpsphoto
import pandas as pd
import numpy as np
# Get the data from image file and return a dictionary
data = {}
for i in range(10):
    data[i] = gpsphoto.getGPSData('provided_image/File_'+"00"+str(i)+'.jpeg')
    data[i]['id'] = i
for i in range(10,36):
    data[i] = gpsphoto.getGPSData('provided_image/File_'+"0"+str(i)+'.jpeg')
    data[i]['id'] = i
df = pd.DataFrame(data)
df = df.T
df.to_csv('latlng.csv', index=False)