#Python script to interpolate the wrf model data.
import os
import re
import xarray as xr
import xesmf as xe
import numpy as np
import gc


parent_path = '/content/drive/MyDrive/Lightning/data/numerical model data/wrf_model.nc'
file_name = '/content/drive/MyDrive/Lightning/data/numerical model data/converted_df.nc'

ds = xr.open_dataset(parent_path).cg_flashco.drop_dims('lev_2')
#da = ds['cg_flashco'][:,0,:,:]
ds_out = xr.Dataset({'latitude': (['latitude'],np.arange(22,14.75,-0.25)), 'longitude': (['longitude'], np.arange(73,81.25,0.25)),})
regridder = xe.Regridder(ds, ds_out, 'nearest_s2d')
dr_out = regridder(ds)
dr_out.to_netcdf(file_name)
print("-- NETCDF FILE CREATED SUCCESSFULLY --")

print("-- STOPPED --")
