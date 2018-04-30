"""
This script generates the maps and density plots from the Chicago datasets.
It uses datasets that can be found in the github repo
"""

import folium as fl
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt



##############################################################
###             This part prepares the datasets            ###
##############################################################


airbnbs_full = pd.read_csv('listings.csv', index_col=0)
crimes_full = pd.read_csv('Crimes_-_2017.csv', index_col=0)
landmarks_full = pd.read_csv('Individual_Landmarks_-_Map.csv', index_col=0)

airbnbs_full = airbnbs_full.rename(columns={'latitude':'Latitude', 'longitude':'Longitude'})
landmarks_full = landmarks_full.rename(columns={'LATITUDE':'Latitude', 'LONGITUDE':'Longitude'})
arrests_full = crimes_full[crimes_full['Arrest']==True]


airbnbs = airbnbs_full.filter(['Latitude','Longitude'], axis=1).dropna()
arrests = arrests_full.filter(['Latitude','Longitude'], axis=1).dropna()
landmarks = landmarks_full.filter(['Latitude','Longitude'], axis=1).dropna()
#parks_c = parks.filter(['Latitude','Longitude'], axis=1).dropna()

datasets = {
    'airbnbs':[airbnbs],
    'arrests':[arrests],
    'landmarks':[landmarks]
}



##############################################################
###   This part generates a map of features using folium   ###
##############################################################

airbnbs_map = airbnbs_full.rename(columns={'latitude':'Latitude', 'longitude':'Longitude',
                                           'review_scores_rating':'Score', 'price':'Price'})
airbnbs_map = airbnbs_map.filter(['Latitude','Longitude','Score','Price'], axis=1).dropna()
arrests_map = arrests_full.filter(['Latitude','Longitude','Primary Type'], axis=1).dropna()
landmarks_map = landmarks_full.filter(['Latitude','Longitude','LANDMARK NAME'], axis=1).dropna()

airbnbs_lat = list(airbnbs_map["Latitude"])
airbnbs_lon = list(airbnbs_map["Longitude"])
airbnbs_sc = list(airbnbs_map["Score"])
airbnbs_pr = list(airbnbs_map["Price"])

crimes_lat = list(arrests_map["Latitude"])
crimes_lon = list(arrests_map["Longitude"])

landmarks_lat = list(landmarks_map["Latitude"])
landmarks_lon = list(landmarks_map["Longitude"])
landmarks_desc = list(landmarks_map.index)


map = fl.Map(location=[41.9,-87.65], zoom_start=12)
fg_a = fl.FeatureGroup(name="Airbnbs")
#fg_c = fl.FeatureGroup(name="Crimes")
fg_l = fl.FeatureGroup(name="Landmarks")


for lt,ln,pr in zip(airbnbs_lat,airbnbs_lon,airbnbs_pr):
    fg_a.add_child(fl.CircleMarker(location=[lt,ln], radius=2, popup=fl.Popup(
    "Airbnb listing \n Price: %s"%pr, parse_html=True), color="blue"))

# for lt,ln in zip(crimes_lat,crimes_lon):
#     fg_c.add_child(fl.Marker(location=[lt,ln], popup=fl.Popup(
#     "Crime",
#     parse_html=True), icon=fl.Icon(color='red')
#     ))

for lt,ln,desc in zip(landmarks_lat,landmarks_lon,landmarks_desc):
    fg_l.add_child(fl.CircleMarker(location=[lt,ln], radius=3, popup=fl.Popup(
    desc, parse_html=True), color="red"))


map.add_child(fg_a)
#map.add_child(fg_c)
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


def kde_map(lon_vec, lat_vec, kernel):
    X, Y = np.meshgrid(lon_vec, lat_vec)
    gridpoints = np.vstack([X.ravel(), Y.ravel()])
    Z = kernel(gridpoints)

    return Z / np.std(Z)

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


lonmin = -88
lonmax = -87.5
latmin = 41.65
latmax = 42.05
npts = 50
bandwidth = 0.3

x = np.linspace(lonmin, lonmax, npts)
y = np.linspace(latmin, latmax, npts)

X, Y = np.meshgrid(x, y, indexing='ij')
positions = np.vstack([X.ravel(), Y.ravel()])


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
ax1_x, ax1_y = datasets['arrests'][0]['Longitude'], datasets['arrests'][0]['Latitude']
ax1.plot(ax1_x, ax1_y, '.', color='red', alpha=0.05, label = 'crimes')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('crimes')

ax2_x, ax2_y = datasets['airbnbs'][0]['Longitude'], datasets['airbnbs'][0]['Latitude']
ax2.plot(ax2_x, ax2_y, '.', color='blue', alpha=0.1, label = 'airbnbs')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('airbnbs')

ax3_x, ax3_y = datasets['landmarks'][0]['Longitude'], datasets['landmarks'][0]['Latitude']
ax3.plot(ax3_x, ax3_y, 'o', color='green', label = 'landmarks')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.set_title('landmarks')

plt.savefig('feature_plots.png')



for key in datasets:
    datasets[key].append(kde_fct(
            datasets[key][0]['Longitude'],
            datasets[key][0]['Latitude'],
            bandwidth))

for key in datasets:
    datasets[key].append(kde_map(x, y, datasets[key][1]))


f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(8,8), sharex=True, sharey=True)
ax1_x, ax1_y = datasets['airbnbs'][0]['Longitude'], datasets['airbnbs'][0]['Latitude']
ax1.plot(ax1_x, ax1_y, '.', color='red', alpha=0.1, label = 'airbnbs')
ax3.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('airbnbs')

ax2_x, ax2_y = datasets['arrests'][0]['Longitude'], datasets['arrests'][0]['Latitude']
ax2.plot(ax2_x, ax2_y, '.', color='blue', alpha=0.01, label = 'crimes')
ax2.set_title('crimes')

ax3_x, ax3_y = datasets['landmarks'][0]['Longitude'], datasets['landmarks'][0]['Latitude']
ax3.plot(ax3_x, ax3_y, 'o', color='green', label = 'landmarks')
ax3.set_title('landmarks')

kde_plot(x, y, datasets['airbnbs'][-1], 'airbnbs', ax4)
kde_plot(x, y, datasets['arrests'][-1], 'crimes', ax5)
kde_plot(x, y, datasets['landmarks'][-1], 'landmarks', ax6)

plt.savefig('feature_heatmaps.png')


map_df = pd.DataFrame({
        'crime':-1*datasets['arrests'][-1],
        'airbnb':-1*datasets['airbnbs'][-1],
        'landmark':datasets['landmarks'][-1],
    })
pred_map = map_df.sum(axis=1).values


plt.figure()
plt.imshow(
        np.rot90(np.reshape(pred_map, (len(x), len(y))).T),
        cmap=plt.cm.RdBu,
        extent=[min(x), max(x), min(y), max(y)]
    )
plt.axis([min(x), max(x), min(y), max(y)])
plt.title('Prediction')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar()

plt.savefig('pred_map.png')
