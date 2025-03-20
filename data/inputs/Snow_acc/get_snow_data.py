import cdsapi

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "snowfall",
        "snowmelt",
        "snow_evaporation"
    ],
    "year": "2025",
    "month": "01",
    "day": ["01"],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()


