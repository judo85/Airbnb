"""
This script generates the maps and density plots from the Chicago datasets.
It uses processed datasets that can be found in the github repo
"""

import folium as fl
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt



##############################################################
###             This part prepares the datasets            ###
##############################################################


airbnbs_full = pd.read_csv('clean_ds/airbnbs_full.csv', index_col=0)
#facilities_full = pd.read_csv('clean_ds/facilities_full.csv', index_col=0)
#groceries_full = pd.read_csv('clean_ds/groceries_full.csv', index_col=0)
crimes_full = pd.read_csv('clean_ds/crimes_full.csv', index_col=0)
landmarks_full = pd.read_csv('clean_ds/landmarks_full.csv', index_col=0)

#groceries_full = groceries_full.rename(columns={'LATITUDE':'Latitude', 'LONGITUDE':'Longitude'})
airbnbs_full = airbnbs_full.rename(columns={'latitude':'Latitude', 'longitude':'Longitude'})
landmarks_full = landmarks_full.rename(columns={'LATITUDE':'Latitude', 'LONGITUDE':'Longitude'})

airbnbs = airbnbs_full.filter(['Latitude','Longitude'], axis=1).dropna()
groceries = groceries_full.filter(['Latitude','Longitude'], axis=1).dropna()
crimes = crimes_full.filter(['Latitude','Longitude'], axis=1).dropna()
landmarks = landmarks_full.filter(['Latitude','Longitude'], axis=1).dropna()
#parks_c = parks.filter(['Latitude','Longitude'], axis=1).dropna()

datasets = {
    'airbnbs':[airbnbs],
    'crimes':[crimes],
    'groceries':[groceries],
    'landmarks':[landmarks]
}



##############################################################
###   This part generates a map of features using folium   ###
##############################################################


airbnbs_lat = list(airbnbs["Latitude"])
airbnbs_lon = list(airbnbs["Longitude"])

crimes_lat = list(crimes["Latitude"])
crimes_lon = list(crimes["Longitude"])

landmarks_lat = list(landmarks["Latitude"])
landmarks_lon = list(landmarks["Longitude"])

map = fl.Map(location=[41.9,-87.65], zoom_start=6, tiles="Mapbox Bright")
fg_a = fl.FeatureGroup(name="Airbnbs")
fg_c = fl.FeatureGroup(name="Crimes")
fg_l = fl.FeatureGroup(name="Landmarks")


for lt,ln in zip(airbnbs_lat,airbnbs_lon):
    fg_a.add_child(fl.Marker(location=[lt,ln], popup=fl.Popup(
    "Airbnb listing",
    parse_html=True)
#    , icon=fl.Icon(color=color_prod(el))
    ))

for lt,ln in zip(crimes_lat,crimes_lon):
    fg_c.add_child(fl.Marker(location=[lt,ln], popup=fl.Popup(
    "Crime",
    parse_html=True)
#    , icon=fl.Icon(color=color_prod(el))
    ))

for lt,ln in zip(landmarks_lat,landmarks_lon):
    fg_c.add_child(fl.Marker(location=[lt,ln], popup=fl.Popup(
    "Landmark",
    parse_html=True)
#    , icon=fl.Icon(color=color_prod(el))
    ))

map.add_child(fg_a)
map.add_child(fg_c)
map.add_child(fg_l)
map.add_child(fl.LayerControl())
map.save("Map.html")




############################################################
###    This part generates kernel density estimates      ###
###    and the prediction heatmap from sample features   ###
############################################################



def kde_fct(lon, lat, bandwidth):

    kernel = gaussian_kde((lon, lat), bw_method=bandwidth)

    return kernel


def kde_plot(lons, lats, kernel_density, mapname, axn):

    axn.imshow(
        np.rot90(np.reshape(kernel_density, (len(lons), len(lats))).T),
        cmap=plt.cm.RdBu,
        extent=[min(lons), max(lons), min(lats), max(lats)]
    )
    axn.axis([min(lons), max(lons), min(lats), max(lats)])
    axn.set_title(mapname)
    axn.set_xlabel('Longitude')
    axn.set_ylabel('Latitude')


def kde_map(lon_vec, lat_vec, kernel):

    X, Y = np.meshgrid(lon_vec, lat_vec)
    gridpoints = np.vstack([X.ravel(), Y.ravel()])
    Z = kernel(gridpoints)

    return Z / np.std(Z)


# compute Gaussian kernel density estimates for all datasets
bandwidth = 0.3
for key in datasets:
    datasets[key].append(compute_kde(
            datasets[key][0]['Longitude'],
            datasets[key][0]['Latitude'],
            bandwidth))


# Boundary conditions for all maps (longitudes as x vals, latitudes as y vals)
lonmin = -88
lonmax = -87.5
latmin = 41.65
latmax = 42.05

# number of points along each map edge
npts = 50

x = np.linspace(lonmin, lonmax, npts)
y = np.linspace(latmin, latmax, npts)

X, Y = np.meshgrid(x, y, indexing='ij')
positions = np.vstack([X.ravel(), Y.ravel()])

for key in datasets:
    datasets[key].append(kde_map(x, y, datasets[key][1]))

map_df = pd.DataFrame({
        'crime':-1*datasets['crimes'][-1],
        'airbnb':-1*datasets['airbnbs'][-1],
        'landmark':datasets['landmarks'][-1],
    })
ALL = rec_map = map_df.sum(axis=1).values

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,7))
f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.5)
kde_plot(x, y, datasets['crimes'][-1], 'Crimes', ax1)
kde_plot(x, y, datasets['airbnbs'][-1], 'Airbnbs', ax2)
kde_plot(x, y, datasets['landmarks'][-1], 'Landmarks', ax3)
kde_plot(x, y, ALL, 'Prediction', ax4)

pylab.savefig('pred_map.png')
