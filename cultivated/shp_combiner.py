import os
import pandas as pd
import geopandas as gpd
folder = "/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2010"
shapefiles = os.listdir(folder)
path = [os.path.join(folder, i) for i in shapefiles if ".shp" in i]
gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in path], ignore_index=True), crs=gpd.read_file(path[0]).crs)
gdf.to_file('/g/data/r78/LCCS_Aberystwyth/training_data/cultivated/2010_merged/2010_merged.shp')
