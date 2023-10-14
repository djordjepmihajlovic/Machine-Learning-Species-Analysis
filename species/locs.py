from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="s2572047@ed.ac.uk")
location = geolocator.reverse("55.92186485133942, -3.1742764925776212")
print(location.address)