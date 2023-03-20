import pickle

import laspy
import numpy as np
import open3d as o3d

from Tiling import points

# load the output from the file
with open("PtCld_Tiles.pkl", "rb") as f:
    Ptcld_Tiles = pickle.load(f)

#
# # Define a function to detect planes from a point cloud using Open3D RANSAC
# def detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations):
#     # Create an empty list to store the plane models
#     planes = []
#     # Create a copy of the point cloud
#     remaining_points = point_cloud.copy()
#     # Loop until no more planes are found or the point cloud is empty
#     while True:
#         # Segment a plane from the point cloud using Open3D RANSAC
#         plane_model, inliers = remaining_points.segment_plane(distance_threshold, ransac_n, num_iterations)
#         # If no inliers are found, break the loop
#         if len(inliers) == 0:
#             break
#         # Extract the inlier points as a new point cloud
#         inlier_cloud = remaining_points.select_by_index(inliers)
#         # Append the plane model and the inlier cloud to the planes list
#         planes.append((plane_model, inlier_cloud))
#         # Remove the inlier points from the remaining point cloud
#         remaining_points = remaining_points.select_by_index(inliers, invert=True)
#     # Return the planes list
#     return planes
#
#
# # Define a dictionary that contains keys for point cloud tiles and arrays of the point cloud data
# point_cloud_dict = {
#     # Example data, you can replace with your own
#     "tile_1": [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
#     "tile_2": [[2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0], [2, 0, 1], [2, 1, 1], [3, 0, 1], [3, 1, 1]],
#     "tile_3": [[0, 2, 0], [0, 3, 0], [1, 2, 0], [1, 3, 0], [0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1]],
#     "tile_4": [[2, 2, 0], [2, 3, 0], [3, 2, 0], [3, 3, 0], [2, 2, 1], [2, 3, 1], [3, 2, 1], [3, 3, 1]]
# }
#
# # Define the parameters for RANSAC
# distance_threshold = 0.01  # Max distance a point can be from the plane model, and still be considered an inlier
# ransac_n = 3  # Number of points to sample as a minimum data set
# num_iterations = 100  # Number of iterations to run
#
# # Loop through the keys of the dictionary and apply the detect_planes function to each array of point cloud data
# for key, value in point_cloud_dict.items():
#     # Convert the array of point cloud data to an Open3D point cloud object
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(value)
#     # Detect the planes from the point cloud using Open3D RANSAC
#     planes = detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations)
#     # Print the results
#     print(f"Planes detected for {key}:")
#     for i, (plane_model, inlier_cloud) in enumerate(planes):
#         print(f"Plane {i + 1}:")
#         print(f"Plane model: {plane_model}")
#         print(f"Number of inlier points: {len(inlier_cloud.points)}")
#         print(f"Inlier points: {inlier_cloud.points}")
#     print()
#
# # faced an error with the copy method using open3d
# # %%
# # Import Open3D library
# import open3d as o3d
#
#
# # Define a function to detect planes from a point cloud using Open3D RANSAC
# def detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations):
#     # Create an empty list to store the plane models
#     planes = []
#     # Create a numpy array from the point cloud points
#     point_cloud_array = np.asarray(point_cloud.points)
#     # Loop until no more planes are found or the point cloud is empty
#     while True:
#         # Segment a plane from the point cloud using Open3D RANSAC
#         plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
#         # If no inliers are found, break the loop
#         if len(inliers) == 0:
#             break
#         # Extract the inlier points as a new point cloud
#         inlier_cloud = point_cloud.select_by_index(inliers)
#         # Append the plane model and the inlier cloud to the planes list
#         planes.append((plane_model, inlier_cloud))
#         # Remove the inlier points from the point cloud array
#         point_cloud_array = np.delete(point_cloud_array, inliers, axis=0)
#         # Create a new point cloud object from the remaining points
#         point_cloud = o3d.geometry.PointCloud()
#         point_cloud.points = o3d.utility.Vector3dVector(point_cloud_array)
#     # Return the planes list
#     return planes
#
#
# # Define a dictionary that contains keys for point cloud tiles and arrays of the point cloud data
# point_cloud_dict = {
#     # Example data, you can replace with your own
#     "tile_1": [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
#     "tile_2": [[2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0], [2, 0, 1], [2, 1, 1], [3, 0, 1], [3, 1, 1]],
#     "tile_3": [[0, 2, 0], [0, 3, 0], [1, 2, 0], [1, 3, 0], [0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1]],
#     "tile_4": [[2, 2, 0], [2, 3, 0], [3, 2, 0], [3, 3, 0], [2, 2, 1], [2, 3, 1], [3, 2, 1], [3, 3, 1]]
# }
#
# # Define the parameters for RANSAC
# distance_threshold = 0.01  # Max distance a point can be from the plane model, and still be considered an inlier
# ransac_n = 3  # Number of points to sample as a minimum data set
# num_iterations = 100  # Number of iterations to run
#
# # Loop through the keys of the dictionary and apply the detect_planes function to each array of point cloud data
# for key, value in Ptcld_Tiles.items():
#     # Convert the array of point cloud data to an Open3D point cloud object
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(value)
#     # Detect the planes from the point cloud using Open3D RANSAC
#     planes = detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations)
#     # Print the results
#     print(f"Planes detected for {key}:")
#     for i, (plane_model, inlier_cloud) in enumerate(planes):
#         print(f"Plane {i + 1}:")
#         print(f"Plane model: {plane_model}")
#         print(f"Number of inlier points: {len(inlier_cloud.points)}")
#         print(f"Inlier points: {inlier_cloud.points}")
#     print()
#
# pcd = o3d.io.read_point_cloud("Input/track1_filtered.ply")
#
# planes = detect_planes(pcd, distance_threshold, ransac_n, num_iterations)
#
# # error with ransac sample points
# # %%
#
# import numpy as np
# from sklearn.linear_model import RANSACRegressor
# from sklearn.decomposition import PCA
#
# # set RANSAC parameters
# ransac = RANSACRegressor(
#     base_estimator=PCA(n_components=3),
#     residual_threshold=0.1,
#     max_trials=1000,
#     stop_probability=0.999,
#     random_state=0
# )
#
#
# # define function to fit plane with RANSAC on point cloud
# def fit_plane_with_ransac(pc_tile):
#     # fit PCA to point cloud
#     pca = PCA(n_components=3)
#     X = pca.fit_transform(pc_tile)
#
#     # apply RANSAC to find inliers for plane model
#     y = np.zeros(X.shape[0])  # dummy y input with same number of rows as X
#
#     ransac = RANSACRegressor(
#         base_estimator=None,
#         residual_threshold=0.1,
#         max_trials=1000,
#         stop_probability=0.999,
#         random_state=0
#     )
#     ransac.fit(X, y)
#     inlier_mask = ransac.inlier_mask_
#
#     # extract inlier points
#     inliers = pc_tile[inlier_mask, :]
#
#     # extract plane parameters from PCA
#     normal = pca.components_[2, :]
#     point = np.mean(inliers, axis=0)
#
#     return (point, normal, inlier_mask)
#
#
# # define function to apply RANSAC to each tile in dictionary
# def apply_ransac_to_dict(pc_dict):
#     # initialize empty dictionary for inliers and planes
#     inlier_dict = {}
#     plane_dict = {}
#
#     # iterate over point cloud tiles
#     for tile_name, pc_tile in pc_dict.items():
#         # check if tile has any points
#         if pc_tile.shape[0] > 0:
#             # fit plane with RANSAC
#             point, normal, inlier_mask = fit_plane_with_ransac(pc_tile)
#
#             # add inliers and plane parameters to dictionaries
#             inlier_dict[tile_name] = inlier_mask
#             plane_dict[tile_name] = (point, normal)
#
#     return (inlier_dict, plane_dict)
#
#
# apply_ransac_to_dict(Ptcld_Tiles)
#
# # error with y argument
# %% applying ransac to an individual tile and extracting the inliers, outliers and plane equation, along with
# the difference in the z values between the inliers and outliers (max z in inliers - min z in outliers)

pcd = o3d.geometry.PointCloud()
pc_array = Ptcld_Tiles[list(Ptcld_Tiles.keys())[2]]
pcd.points = o3d.utility.Vector3dVector(pc_array)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
# now I have the plane equation in plane model, and the points inside within the plane in inliers
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1.0, 0, 0])  # inliers in red
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # outliers in grey
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

# from sklearn.metrics import pairwise_distances
# import numpy as np
# from scipy.spatial.distance import cdist
#
# # this doesn't take into account the z axis
# inlier_z = np.asarray(inlier_cloud.points)[:, 2]
# outlier_z = np.asarray(outlier_cloud.points)[:, 2]
# # distances = cdist(np.asarray(inlier_cloud.points), np.asarray(outlier_cloud.points))
# # max_distance = np.max(distances)
# np.average(inlier_z) - np.average(outlier_z)  # x10 for cm, x100 mm: 13 mm difference
#
# np.max(inlier_z)- np.min(inlier_z)

import numpy as np

#this calculates the maximum z distance between the plane and all outlier points
def max_distance_to_plane(points, plane):
    """
    Calculate the maximum distance between a plane and a set of points in the z-direction.
    """

    def distance_to_plane(plane, point):
        """
        Calculate the distance between a plane and a point in the z-direction.
        """
        return np.abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3])

    distances = [distance_to_plane(plane, point) for point in points]
    return max(distances)


# Example usage

max_distance = max_distance_to_plane(pc_array, plane_model)
print(f"Maximum distance: {max_distance}")

#%%

# this calculates the maximum z distance between the plane and all outlier points below the plane
def max_distance_to_plane_below(points, plane):
    """
    Calculate the maximum distance between a plane and a set of points below the plane in the z-direction.
    """
    def distance_to_plane(plane, point):
        """
        Calculate the distance between a plane and a point in the z-direction.
        """
        distance = np.abs(plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3])
        if plane[2] > 0 and point[2] > -plane[3]/plane[2]:
            return 0
        return distance

    distances = [distance_to_plane(plane, point) for point in points if plane[2] > 0 and point[2] <= -plane[3]/plane[2]]
    return max(distances)


max_distance_below = max_distance_to_plane_below(pc_array, plane_model)
print(f"Maximum distance below: {max_distance_below}")

#both methods give the same result in this case. I think the second method is more accurate

#69 cm is too high. it is illogical

#$$ lets try to export the single file as las so we need to match it with the original las file to get the columns



#
# import numpy as np
#
# from LAS_tools import _las_to_df
#
# las_file=laspy.read("Input/track2_pc.las")
# df=_las_to_df(las_file)
#
#
# df_array = df.to_numpy()
# # create example arrays
# x = pc_array
# y = df_array
#
# # find indices of matching rows
# matching_rows_idx = np.where(np.isin(y[:, 0], x[:, 0]))[0]
#
# column_names = np.array(['x','y','z','intensity', 'return_number', 'number_of_returns',
#                                         'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel',
#                                         'scan_direction_flag', 'edge_of_flight_line', 'classification',
#                                         'user_data', 'scan_angle', 'point_source_id', 'gps_time', 'red',
#                                         'green', 'blue'])
#
# # extract columns for matching rows
# matching_cols_y = y[matching_rows_idx, :]
# matching_col_names = column_names
#
# # concatenate column names with matching columns
# matching_cols_y = np.vstack((matching_col_names, matching_cols_y))
#
# # print the original and new arrays
# print("Array x:\n", x)
# print("Array y:\n", y)
# print("Matching columns in array y:\n", matching_cols_y)

#they are not matching properly, ask afshin.
#lets try to export only the points that are in the tile and see if it works

import pandas as pd

df = pd.DataFrame(data=pc_array, columns=['x','y','z'])

from manipulation_tools import _df_to_las_conversion

_df_to_las_conversion(df, address='Input', name='Test2',
                      data_columns=['x', 'y', 'z',])
#tile1 overlaps with the original las file, so it is working and placed correctly.


#%% loop over all tiles and export them as las files

import open3d as o3d
import numpy as np
import pandas as pd
from manipulation_tools import _df_to_las_conversion

# Loop over point clouds
for i in range(len(Ptcld_Tiles)):
    # Get point cloud array
    pc_array = Ptcld_Tiles[list(Ptcld_Tiles.keys())[i]]

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)

    # Segment plane
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # Draw point cloud with inliers in red and outliers in grey
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # inliers in red
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # outliers in grey
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # Calculate maximum distance to plane
    max_distance = max_distance_to_plane(pc_array, plane_model)
    print(f"Maximum distance: {max_distance}")

    # Convert array to pandas DataFrame
    df = pd.DataFrame(data=pc_array, columns=['x', 'y', 'z'])

    # Convert DataFrame to LAS file
    name = f'TIle{i + 1}'
    _df_to_las_conversion(df, address='Input', name=name, data_columns=['x', 'y', 'z'])


for tile_name in Ptcld_Tiles.keys():
    pc_array = Ptcld_Tiles[tile_name]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # inliers in red
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # outliers in grey

    max_distance = max_distance_to_plane(pc_array, plane_model)
    print(f"Maximum distance for tile {tile_name}: {max_distance}")

    df = pd.DataFrame(data=pc_array, columns=['x','y','z'])

    _df_to_las_conversion(df, address='Input', name=f'{tile_name}_output',
                          data_columns=['x', 'y', 'z'])

