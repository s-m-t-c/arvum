import xarray as xr
from datacube.utils.geometry import assign_crs
from odc.algo import xr_reproject
from pathlib import Path
from datacube.utils.cog import write_cog
folder = "/g/data/zv2/agcd/v1/precip/total/r005/01month/*.nc"
file_list = []
#for path in Path(folder).rglob('*.nc'):
#    data = xr.open_rasterio(path)
#    data = data.sum('band')
#    print(data)
#    file_list.append(data)
#    break

data = xr.open_mfdataset(folder, combine="by_coords")
#print(data)
#print(data['precip'])
out = data['precip'].sum('time').compute()

out.to_netcdf("agdc_all_time.nc")
