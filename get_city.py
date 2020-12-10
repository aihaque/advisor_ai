from geopy import Nominatim

geolocator = Nominatim(user_agent = 'test_1')
cor = ('26.8393','80.9231')
location = geolocator.reverse(cor)
l = location.raw['address']
print (l['city'])
cor2 = ('49.2781019921437','-122.922451786348')
location2 = geolocator.reverse(cor2)
l = location2.raw['address']
print (l['city'])