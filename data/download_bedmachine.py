import earthaccess

auth = earthaccess.login(persist=True)

data = 'BedMachineGreenland-v5.nc'

"""
username: karlfindhansen
password: vyfcaw-Nyrmy2-dyxvid
"""

url_nc_file = "https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/1993.01.01/BedMachineGreenland-v5.nc"
url_tif_file = "https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/1993.01.01/BedMachineGreenland-v5_bed.tif"

local_tif_filename = 'BedMachineGreenland-v5_bed.tif'
local_nc_filename = "data/BedMachineGreenland-v5.nc"

session = earthaccess.get_requests_https_session()
headers = {"Range": "bytes=0-100"}
r = session.get(url_nc_file, headers=headers)

fs = earthaccess.get_fsspec_https_session()

def download_file(url, local_filename):
    print(f"Downloading {local_filename}")
    with fs.open(url, "rb") as remote_file:
        with open(local_filename, "wb") as local_file:
            local_file.write(remote_file.read())  
    print(f"Download complete: {local_filename}")

download_file(url_nc_file, local_nc_filename)
download_file(url_tif_file, local_tif_filename)
