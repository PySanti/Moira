from utils.build_climate_data import get_weather_features

data = get_weather_features('new york','11-06-10')

for k,v in data.items():
    print(k,v)

