import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import laspy
from src.PLY_tools import _ply_to_df
from pyntcloud import PyntCloud

from src.LAS_tools import _las_to_df
from src.manipulation_tools import _df_to_las_conversion
pcd = o3d.io.read_point_cloud(
    'D:/Projects/PC_Rutting/Input/track2_pc.ply')
las_file= laspy.read("../Input/track2_pc.las") #didnt use it
ply_file = PlyData.read("../Input/track2_pc.ply")
track2_pc_df2= _ply_to_df(ply_file)
track2_pc_df= _las_to_df(las_file) #didnt use it
# Get the current working directory
cwd = os.getcwd()
cwd


with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

#need to check the labels and filter the main las file according to the desired road cluster
unique, counts = np.unique(labels, return_counts=True)
labels
# Combine unique values and their counts into a dictionary
counts_dict = dict(zip(unique, counts))
print(counts_dict)
#label 0 road. apply boolean filter

value = 0
bool_array = (labels == value)

track2_pc_dbscan = track2_pc_df2[bool_array]


track2_pc_dbscan.shape[0] - track2_pc_df2.shape[0]  # ok


cloud = PyntCloud(track2_pc_dbscan)
cloud.to_file("../Input/track2_pc_dbscan.ply") #scan turned out bad. might be because I removed other points as well, will need to experiment with this depending on different bolean values

# _df_to_las_conversion(track3_filtered,
#                       address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
#                       name='track3_filtered', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
#         , 'red', 'green', 'blue'])