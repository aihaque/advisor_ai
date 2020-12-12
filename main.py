import pandas as pd
import numpy as np
import math
import mplleaflet
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats
import seaborn
seaborn.set()

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
        print("Looking for following of categorical interests")
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

    if(joined.empty):
        print("Sorry nothing to suggest")
    else:
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
            f = open("Your trip suggessstion.txt", "w")
            f.write("I suggest you could go for the uploaded trip to the following amenities\n")
            print("I suggest you could go for the uploaded trip to the following amenities ")
            
            suggested = suggested.reset_index()
            p = suggested.drop(columns=['lat','lon'])
            print(p)
            try:
                f.write(p.to_string())
            except KeyError:
                f.write("Null")
            f.close()

            plt.figure(figsize=(8,6))
            fig = plt.figure()
            #x = suggested[suggested['amenity'] == 'atm']
            #x = x.reset_index()
            #plt.scatter(x['lon'],x['lat'])

            plt.scatter(suggested['lon'],suggested['lat'])

            filtered = lowess(suggested['lat'],suggested['lon'], frac=0.2)
            plt.plot(filtered[:, 0], filtered[:, 1],'r-', linewidth=5)

            #plt.plot(x,y, 'r-', linewidth=3)
            print("Please see the map with blue dot is the suggessted amenities red line is your optimal waliking")
            print() 
            mplleaflet.show(fig=fig)


def execute_nightclub_AI(nightclub,amenities):

    joined = nightclub.join(amenities.set_index('city'),lsuffix='_caller',on='city')

    if(joined.empty):
        print("Sorry no popular nightclub found in your city")
    else:
        joined['distance']  = joined.apply(distance,axis=1)
        joined = joined[joined['distance']<1000]

        joined['count'] = joined.groupby(['lat_caller','lon_caller'])['tags'].transform('count')
        joined = joined[joined['count']>10]
        if(joined.empty):
            print("Sorry no popular nightclub found in your city")
        else:
            suggested = pd.DataFrame()
            suggested = joined
            suggested = suggested.drop_duplicates(subset=['lat_caller','lon_caller'])

            f = open("Nightclub suggessstion.txt", "w")
            f.write('Suggested Nightclub info you can find more than 10 Nightclub related amenities within 1Km\n')
            print('Suggested Nightclub info you can find more than 10 Nightclub related amenities within 1Km')
            print(suggested['address_caller'])

            try:
                f.write(suggested['address_caller'].to_string())
            except KeyError:
                f.write("Null")

            f.close()

            long_info = pd.DataFrame()
            long_info['Suggested NightClubs info'] = joined['address_caller']
            long_info['Realated_Aminities_Within_1Km'] = joined['amenity']

            print()
            print("Related amenities info")
            print(long_info)

            plt.figure(figsize=(8,6))
            fig = plt.figure()
            plt.scatter(joined['lon_caller'],joined['lat_caller'],30)
            plt.scatter(joined['lon'],joined['lat'],10)

            filtered = lowess(joined['lat_caller'],joined['lon_caller'], frac=0.7)
            plt.plot(filtered[:, 0], filtered[:, 1],'r-', linewidth=5)
            print("Please see the map with blue dot is the suggessted nightclub and orange dot is the nearby related amenities red line is your optimal waliking")
            print() 
            mplleaflet.show(fig=fig)

def execute_restaurant_AI(restaurant,amenities):

    joined = restaurant.join(amenities.set_index('city'),lsuffix='_caller',on='city')

    if(joined.empty):
        print("Sorry no popular restaurant found in your city")
    else:
        joined['distance']  = joined.apply(distance,axis=1)
        joined = joined[joined['distance']<500]

        joined['count'] = joined.groupby(['lat_caller','lon_caller'])['tags'].transform('count')
        joined = joined[joined['count']>40]
        if(joined.empty):
            print("Sorry no popular restaurant found in your city")
        else:
            suggested = pd.DataFrame()
            suggested = joined
            suggested = suggested.drop_duplicates(subset=['lat_caller','lon_caller'])

            f = open("restuarant suggessstion.txt", "w")
            f.write('Suggested restaurant info you can find more than 40 restaurant related amenities within 500meter\n')
            print(suggested['address_caller'])

            try:
                f.write(suggested['address_caller'].to_string())
            except KeyError:
                f.write("Null")

            f.close()

            long_info = pd.DataFrame()
            long_info['Suggested restaurant info'] = joined['address_caller']
            long_info['Chain_restaurants_Within_500meter'] = joined['amenity']
            print()
            print("related amenities info")
            print(long_info)

            plt.figure(figsize=(8,6))
            fig = plt.figure()
            plt.scatter(joined['lon_caller'],joined['lat_caller'],30)
            plt.scatter(joined['lon'],joined['lat'],10)

            filtered = lowess(joined['lat_caller'],joined['lon_caller'], frac=0.7)
            plt.plot(filtered[:, 0], filtered[:, 1],'r-', linewidth=5)
            print("Please see the map with blue dot is the suggessted resturant and orange dot is the nearby related amenities red line is your optimal waliking")
            print() 
            mplleaflet.show(fig=fig)

def main():
    avg_latlon = pd.read_csv('avg_latlon.csv')

    cities_visted = avg_latlon['city'].unique()
    amenities = []
    print("You have visted follwing cities")
    for city in cities_visted:
        print(city)
        df = pd.read_csv('cities/'+ city + '.csv')
        amenities.append(df)
    amenities = pd.concat(amenities)  
    amenities = user_input(amenities)
    if(amenities.empty or avg_latlon.empty):
        print("Sorry not any popular amenities nearby")
    else:
        print("Trying to suggest some amenities of your interest you would have visited")
        executeAI(avg_latlon,amenities)

    print() 
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
    print()   
    print("I might help you with one more adviced if you will be planning to go a restaurant")
    print("0 Not interested")
    print("1 Interested")
    value = input("")
    value = int(value)
    if(value==1):
        city = city_input()
        amenities = pd.read_csv('cities/'+ city + '.csv')

        restaurant = amenities[amenities['amenity'] == 'restaurant']

        df = pd.read_csv('categories.csv')
        category = df['food'].dropna()
        amenities = amenities[amenities['amenity'].isin(category)]
        if(amenities.empty or restaurant.empty):
            print("Sorry no popular resturant place found in your city")
        else:
            execute_restaurant_AI(restaurant,amenities)
if __name__ == '__main__':
    main()