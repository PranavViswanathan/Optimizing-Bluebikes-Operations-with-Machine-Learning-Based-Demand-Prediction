from NOAA_DataAcq import NOAA


noaa = NOAA()

noaa.fetch_data_from_api()
noaa.get_weather_dataframe()


