from utils.build_climate_data import get_weather_features

data = get_weather_features('new york','23-01-26')

for k,v in data.items():
    print(k,v)

