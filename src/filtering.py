import pandas as pd
import numpy as np
import laspy
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d

from src.manipulation_tools import _df_to_las_conversion

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def _las_to_df(las_file: laspy.LasData) -> pd.DataFrame:
    data = []
    columns = list(las_file.point_format.dimension_names)
    columns[:3] = ['x','y','z']
    for column in columns:
        data.append(np.array(las_file[column]))
    df = pd.DataFrame(np.array(data).T, columns=columns)
    return df


# compressing a LAS file into LAZ and vise versa
def _las_to_laz_conversion(las_file: laspy.LasData, address: str, name: str, do_compress: bool = True):
    if do_compress:
        las_file.write(do_compress=do_compress, destination=Path(f'{address}') / f'{name}.laz')
    else:
        las_file.write(do_compress=do_compress, destination=Path(f'{address}') / f'{name}.las')


if __name__ == '__main__':
    las_file = laspy.read(
        'Ledcor-Highway32/Track_H_20221114_192938 Profiler.zfs_0_annotated.laz')
    track1 = _las_to_df(las_file)
    # df = pd.read_csv('../data/testdf.csv')
    print(track1)
    # _write_df_to_csv(df, '../data/', 'ttlaz')
    # _laz_to_las_conversion(las_file, '../data/', 'ttlas')
    # _df_to_las_conversion(df, address='../data/', name='ttlas3',
    #                       data_columns=['X', 'Y', 'Z', 'intensity', 'classification'])

np.max(track1['z'])

track1_elevation = track1[(track1['z'] > 736) & (track1['z'] <= 745)]

_las_to_laz_conversion(track1_elevation, address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master/Filtered Tracks', name='track1_elevation1', do_compress=True)

_df_to_las_conversion(track1_elevation,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track1_elevation1', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

##DBSCAN

# from sklearn.cluster import DBSCAN
#
# # Load the point cloud data into a numpy array
# points = df2_elevation2.to_numpy()
#
# points = points[0:10000,0:3]
#
#
# # Apply DBSCAN to the point cloud data
# dbscan = DBSCAN(eps=0.3, min_samples=17).fit(points)
#
# # Retain only the road segment clusters
# road_segment_labels = set(dbscan.labels_[dbscan.labels_ != -1])
#
# road_segment_labels= np.array(dbscan.labels_)
# # Create a new point cloud with only the road segments
# road_segment_points = np.array([points[i] for i in range(points.shape[0]) if dbscan.labels_[i] in road_segment_labels])
#
# # Save the road segment point cloud to a file
# np.savetxt('road_segment_points.txt', road_segment_points)


pcd = o3d.io.read_point_cloud(
    '/Filtered Tracks/track1_elevation.ply')

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=2, min_points=50, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

#need to check the labels and filter the main las file according to the desired road cluster
unique, counts = np.unique(labels, return_counts=True)

# Combine unique values and their counts into a dictionary
counts_dict = dict(zip(unique, counts))

#label 14 road. apply boolean filter

value = 14
bool_array = (labels == value)

track1_filtered = track1_elevation[bool_array]


track1_filtered.shape[0] - track1_elevation.shape[0]  # ok


_df_to_las_conversion(track1_filtered,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track1_filtered', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

# pcd2 = o3d.io.read_point_cloud(
#     'C:/Users/unbre\OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master/track1_filtered.ply')

#2nd iteration of the same filtered track -> produces just one cluster anyway
#
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         pcd2.cluster_dbscan(eps=2, min_points=20, print_progress=True))
#
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd2])

las_file = laspy.read( 'C:/Users/unbre/OneDrive - UBC/Ph.D. Work/PCTool/PointCloudTool-master/Ledcor-Highway32/Track_H_20221114_192938 Profiler.zfs_0_annotated.laz')
track1 = _las_to_df(las_file)
    # df = pd.read_csv('../data/testdf.csv')
print(track1)
    # _write_df_to_csv(df, '../data/', 'ttlaz')
    # _laz_to_las_conversion(las_file, '../data/', 'ttlas')
    # _df_to_las_conversion(df, address='../data/', name='ttlas3',
    #                       data_columns=['X', 'Y', 'Z', 'intensity', 'classification'])

track1_elevation = track1[(track1['Z'] > 6900) & (track1['Z'] <= 7200)]



#%% track 2

las_file2 = laspy.read(
    'Ledcor-Highway32/Track_H_20221114_192938 Profiler.zfs_1_annotated.laz')
track2 = _las_to_df(las_file2)
# df = pd.read_csv('../data/testdf.csv')
print(track2)

elevation2 = track2['Z']
elevation2.describe()

elevation1 = track1['Z']
elevation1.describe()

track2_elevation = track2[(track2['Z'] > 3500) & (track2['Z'] <= 5000)]



_df_to_las_conversion(track2_elevation,
                      address='C:/Users/unbre\OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track2_elevation', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

pcd2 = o3d.io.read_point_cloud(
    '/Filtered Tracks/track2_elevation.ply')

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd2.cluster_dbscan(eps=2, min_points=50, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd2.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd2])

#need to check the labels and filter the main las file according to the desired road cluster
unique, counts = np.unique(labels, return_counts=True)

# Combine unique values and their counts into a dictionary
counts_dict = dict(zip(unique, counts))
print(counts_dict)
#label 15 road. apply boolean filter

value = 15
bool_array = (labels == value)

track2_filtered = track2_elevation[bool_array]


track2_filtered.shape[0] - track2_elevation.shape[0]  # ok


_df_to_las_conversion(track2_filtered,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track2_filtered', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])


#%% track 3

las_file3 = laspy.read( 'Ledcor-Highway32/Track_H_20221114_192938 Profiler.zfs_2_annotated.laz')
track3 = _las_to_df(las_file3)
# df = pd.read_csv('../data/testdf.csv')
print(track3)

elevation3 = track3['Z']
elevation3.describe()

track3_elevation = track3[(track3['Z'] > 3000) & (track3['Z'] <= 4000)]



_df_to_las_conversion(track3_elevation,
                      address='C:/Users/unbre\OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track3_elevation', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

pcd3 = o3d.io.read_point_cloud(
    'track3_elevation.ply')

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd3.cluster_dbscan(eps=2, min_points=50, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd3.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd3])

#need to check the labels and filter the main las file according to the desired road cluster
unique, counts = np.unique(labels, return_counts=True)

# Combine unique values and their counts into a dictionary
counts_dict = dict(zip(unique, counts))
print(counts_dict)
#label 15 road. apply boolean filter

value = 5
bool_array = (labels == value)

track3_filtered = track3_elevation[bool_array]


track3_filtered.shape[0] - track3_elevation.shape[0]  # ok


_df_to_las_conversion(track3_filtered,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track3_filtered', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

#%% track 4

las_file4 = laspy.read( 'Ledcor-Highway32/Track_H_20221114_192938 Profiler.zfs_3_annotated.laz')
track4 = _las_to_df(las_file4)
# df = pd.read_csv('../data/testdf.csv')
print(track4)

elevation4 = track4['Z']
elevation4.describe()

track4_elevation = track4[(track4['Z'] > 5500) & (track4['Z'] <= 7000)]



_df_to_las_conversion(track4_elevation,
                      address='C:/Users/unbre\OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track4_elevation', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

pcd4 = o3d.io.read_point_cloud(
    'Filtered Tracks/track4_elevation.ply')

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd4.cluster_dbscan(eps=2, min_points=50, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd4.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd4])

#need to check the labels and filter the main las file according to the desired road cluster
unique, counts = np.unique(labels, return_counts=True)

# Combine unique values and their counts into a dictionary
counts_dict = dict(zip(unique, counts))
print(counts_dict)
#label 15 road. apply boolean filter

value = 5
bool_array = (labels == value)

track4_filtered = track4_elevation[bool_array]


track4_filtered.shape[0] - track4_elevation.shape[0]  # ok


_df_to_las_conversion(track4_filtered,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track4_filtered', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])


#%% segmentation track 4

pcd5 = o3d.io.read_point_cloud("track4_elevation.ply")
plane_model, inliers = pcd5.segment_plane(distance_threshold=1,
                                         ransac_n=5,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd5.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd5.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])