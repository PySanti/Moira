from utils.build_climate_data import get_weather_features

data = get_weather_features('new york','05-12-25')

for k,v in data.items():
    print(k,v)

