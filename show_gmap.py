import matplotlib.pyplot as plt
import mplleaflet
import numpy as np
import pandas as pd
plt.clf()
df = pd.read_csv('avg_latlon.csv')
plt.figure(figsize=(8,6))
fig = plt.figure()

for row in df.head().itertuples():
    plt.scatter(row.lon,row.lat)
    plt.annotate(row.city,(row.lon,row.lat))

plt.show()
#plt.text(df['Longitude'] , df['Latitude'] ,'This text starts at point (2,4)')
mplleaflet.show(fig=fig)


#plt.scatter(2,3)
#plt.annotate('hello',(2,3)) # this is the point to label # horizontal alignment can be left, right or center
#plt.show()