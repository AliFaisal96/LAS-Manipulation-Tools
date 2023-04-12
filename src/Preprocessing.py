import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import laspy
from src.PLY_tools import _ply_to_df
from pyntcloud import PyntCloud
from plyfile import PlyData

from LAS_tools import _las_to_df
from manipulation_tools import _df_to_las_conversion
#
# pcd = o3d.io.read_point_cloud(
#     'Input/track2_pc.ply')
# # las_file= laspy.read("../Input/track2_pc.las") #didnt use it
# ply_file = PlyData.read("Input/track2_pc.ply")
# track2_pc_df2 = _ply_to_df(ply_file)
# # track2_pc_df= _las_to_df(las_file) #didnt use it
# # Get the current working directory
# cwd = os.getcwd()
# cwd
#
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         pcd.cluster_dbscan(eps=0.1, min_points=10, print_progress=True))
#
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
#
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd])
#
# # need to check the labels and filter the main las file according to the desired road cluster
# unique, counts = np.unique(labels, return_counts=True)
# labels
# # Combine unique values and their counts into a dictionary
# counts_dict = dict(zip(unique, counts))
# print(counts_dict)
# # label 0 road. apply boolean filter
#
# value = 0
# bool_array = (labels == value)
#
# track2_pc_dbscan = track2_pc_df2[bool_array]
#
# var = track2_pc_dbscan.shape[0] - track2_pc_df2.shape[0]  # ok
#
# cloud = PyntCloud(track2_pc_dbscan)
# cloud.to_file(
#     "Input/track2_pc_dbscan.ply")
#
# # scan turned out bad. might be because I removed other points as well, will need to experiment with this depending on different bolean values
#
# # %% lets try doing this using numpy
#
# import open3d as o3d
# import numpy as np
#
# # Read in the point cloud from a .ply file
# pcd = o3d.io.read_point_cloud("Input/track2_pc.ply")
#
# # Get the array of points
# points = np.asarray(pcd.points)
#
# points[:, 2].max()
# points[:, 2].min()
#
# # assume that you want to remove points that are very far away from the rest of the point cloud
# max_distance_threshold = 0.23
#
# # calculate the centroid of the point cloud
# centroid = np.mean(points, axis=0)
#
# # calculate the distance of each point in the point cloud from the centroid
#
# distances = np.abs(points[:, 2] - centroid[2])
#
# # identify the points that are very far away from the rest of the point cloud
# outlier_indices = np.where(distances > max_distance_threshold)[0]
#
# # remove the outliers from the point cloud
# cleaned_point_cloud = np.delete(points, outlier_indices, axis=0)
#
# # Remove the points with the specified indices
# inlier_cloud = pcd.select_by_index([i for i in range(len(pcd.points)) if i not in outlier_indices])
#
# # Save the resulting point cloud to a new ply file
# o3d.io.write_point_cloud("Input/outlier_removed_ply_file.ply", inlier_cloud)
#
# # this method is removing ALL points above the centroid of the point cloud by a certain threshold. I need something that removes points
# # that are sparse and not part of the road. I will need to experiment with this
#
#
# #%% grid method
#
# # Set the minimum density threshold (in points per square meter)
# min_density = 100
#
# # Calculate the bounding box of the point cloud
# bbox = pcd.get_axis_aligned_bounding_box()
# min_bound = bbox.min_bound
# max_bound = bbox.max_bound
#
# # Divide the bounding box into a grid of equal-sized cells
# cell_size = 0.1 # meters
# x_cells = int(np.ceil((max_bound[0] - min_bound[0]) / cell_size))
# y_cells = int(np.ceil((max_bound[1] - min_bound[1]) / cell_size))
# z_cells = int(np.ceil((max_bound[2] - min_bound[2]) / cell_size))
# grid = np.zeros((x_cells, y_cells, z_cells), dtype=np.int32)
#
# # For each point, increment the corresponding cell in the grid
# points = np.asarray(pcd.points)
# indices = np.floor((points - min_bound) / cell_size).astype(np.int32)
# for i in range(points.shape[0]):
#     x, y, z = indices[i]
#     if x >= 0 and x < x_cells and y >= 0 and y < y_cells and z >= 0 and z < z_cells:
#         grid[x, y, z] += 1
#
# # Calculate the density of points in each cell
# cell_areas = cell_size ** 2
# densities = grid.astype(np.float32) / cell_areas
#
# # Remove all cells with density below the threshold
# indices = np.argwhere(densities < min_density)
# for i, j, k in indices:
#     grid[i, j, k] = 0
#
# # Merge the remaining cells into a single point cloud
# indices = np.argwhere(grid > 0)
# points = np.empty((indices.shape[0], 3), dtype=np.float32)
# for i in range(indices.shape[0]):
#     x, y, z = indices[i]
#     points[i] = [min_bound[0] + (x + 0.5) * cell_size,
#                  min_bound[1] + (y + 0.5) * cell_size,
#                  min_bound[2] + (z + 0.5) * cell_size]
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
#
# # Save the cleaned point cloud to file
# o3d.io.write_point_cloud('output.ply', pcd)
#
# #takes a long time, removes too many points.
# %% Using KNN to remove sparse points

import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
print(os.getcwd())

pcd = o3d.io.read_point_cloud("../Input/Track 1/track1_pc_cleaned.ply")
point_cloud = np.asarray(pcd.points)

radius = 0.01
threshold = 0.005


nbrs = NearestNeighbors(algorithm='ball_tree').fit(point_cloud)

distances, _ = nbrs.radius_neighbors(point_cloud, radius=radius)
avg_distances = [np.mean(d) if len(d) > 0 else 0 for d in distances]
mask = np.array(avg_distances) < threshold



np.count_nonzero(mask)  # its removing the sparse points.

selected_points = pcd.select_by_index(np.where(mask)[0])
o3d.io.write_point_cloud("../Input/Track 1/track1_kmeans.ply", selected_points)

#works in removing the sparse points. however, decreasing the threshold removes many points. Here im using the KNN algorithm
#from scikit learn to remove the sparse points by calculating the average distance of each point from its 5 nearest neighbors
#and removing the points that are far away from the rest of the point cloud. I use the mask to remove those sparse points from my
#original ply file.

