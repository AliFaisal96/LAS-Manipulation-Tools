import open3d as o3d
import numpy as np

import os

os.chdir("C:/Users/unbre\OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master")

# Load the point cloud
pcd = o3d.io.read_point_cloud("Input/track2_pc.ply")

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)


# Define the tile size

# # Determine the minimum and maximum values for each dimension
# min_x, min_y, min_z = np.min(points, axis=0)
# max_x, max_y, max_z = np.max(points, axis=0)
#
# # Compute the number of tiles along each dimension
# num_tiles_x = int(np.ceil((max_x - min_x) / tile_size))
# num_tiles_y = int(np.ceil((max_y - min_y) / tile_size))
# num_tiles_z = int(np.ceil((max_z - min_z) / tile_size))
#
# # Create a list to store the tiled point clouds
# tiled_point_clouds = []
#
# # Iterate over each tile
# for i in range(num_tiles_x):
#     for j in range(num_tiles_y):
#         for k in range(num_tiles_z):
#             # Compute the bounds of the tile
#             x_min = min_x + i * tile_size
#             x_max = x_min + tile_size
#             y_min = min_y + j * tile_size
#             y_max = y_min + tile_size
#             z_min = min_z + k * tile_size
#             z_max = z_min + tile_size
#
#             print('working...',{i},{j},{k})
#
#             # Extract the points that are within the bounds of the tile
#             indices = np.where(
#                 (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
#                 (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
#                 (points[:, 2] >= z_min) & (points[:, 2] < z_max)
#             )[0]
#
#             # Create a point cloud for the current tile
#             tile = o3d.geometry.PointCloud()
#             tile.points = o3d.utility.Vector3dVector(points[indices])
#
#             # Add the point cloud for the current tile to the list
#             tiled_point_clouds.append(tile)
#
# o3d.visualization.draw_geometries(tiled_point_clouds)
#
# (np.asarray(tiled_point_clouds))

# %% tiling without open3d


def tile_point_cloud(points, tile_size):
    # Determine the minimum and maximum values for each dimension
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    # Compute the number of tiles along each dimension
    num_tiles_x = int(np.ceil((max_x - min_x) / tile_size))
    num_tiles_y = int(np.ceil((max_y - min_y) / tile_size))
    num_tiles_z = int(np.ceil((max_z - min_z) / tile_size))

    # Create a dictionary to store the tiled point clouds
    tiled_point_clouds = {}

    # Iterate over each tile
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            for k in range(num_tiles_z):
                # Compute the bounds of the tile
                x_min = min_x + i * tile_size
                x_max = x_min + tile_size
                y_min = min_y + j * tile_size
                y_max = y_min + tile_size
                z_min = min_z + k * tile_size
                z_max = z_min + tile_size

                print('working...', {i}, {j}, {k})

                # Extract the points that are within the bounds of the tile
                indices = np.where(
                    (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] < z_max)
                )[0]

                # Store the points for the current tile in the dictionary
                tile_key = (i, j, k)
                tiled_point_clouds[tile_key] = points[indices]

    return tiled_point_clouds


result = tile_point_cloud(points, 10)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] != 0}

pc_tile_keys = np.array(list(result.keys()))

result.values()
result.keys()

pc_tile_pts = np.vstack(list(result.values()))
np.shape(pc_tile_pts)
for key in result:
    print(f"{key}: {len(result[key])} values")

import pickle
# save the output to a file called output.pkl
with open("Input/PtCld_Tiles.pkl", "wb") as f:
    pickle.dump(result, f)
# load the output from the file
with open("output.pkl", "rb") as f:
    output = pickle.load(f)


# sum_count = 0 for key, value in result.items(): sum_count += len(value) print(sum_count)
# I was checking if the sum
# of the points in the tiles is equal to the total number of points in the original point cloud I was also checking
# if each array in a key is a number of points, which is the case. so each array holds the points of a tile,
# and the key is the tile number Now I have the keys as tiles, and the values as the points in each tile, and I want
# to save each tile as a ply file? I want to extract the rest of information from the original point cloud,
# like the color, and the normals, and save them in the ply file as well as I can do RANSAC on each tile
# iteratively, and get the points outside the fitting plane

# TODO: see ransac for a single tile
# TODO: iterate over keys and values ransac
# TODO: find a way to optimize the tiling scheme to have consistent number of points in each tile
# TODO: RANSAC ON EACH TILE

#%% All together

import pandas as pd
from manipulation_tools import _df_to_las_conversion
import open3d as o3d
import numpy as np

# Load the point cloud
pcd = o3d.io.read_point_cloud("Input/track2_pc_kmeans.ply")

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)

def tile_point_cloud(points, tile_size):
    # Determine the minimum and maximum values for each dimension
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    # Compute the number of tiles along each dimension
    num_tiles_x = int(np.ceil((max_x - min_x) / tile_size))
    num_tiles_y = 1
    num_tiles_z = int(np.ceil((max_z - min_z) / tile_size))

    # Create a dictionary to store the tiled point clouds
    tiled_point_clouds = {}

    # Iterate over each tile
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            for k in range(num_tiles_z):
                # Compute the bounds of the tile
                x_min = min_x + i * tile_size
                x_max = x_min + tile_size
                y_min = min_y
                y_max = max_y
                z_min = min_z + k * tile_size
                z_max = z_min + tile_size

                print('working...', {i}, {j}, {k})

                # Extract the points that are within the bounds of the tile
                indices = np.where(
                    (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                    (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
                    (points[:, 2] >= z_min) & (points[:, 2] < z_max)
                )[0]

                # Store the points for the current tile in the dictionary
                tile_key = (i, j, k)
                tiled_point_clouds[tile_key] = points[indices]

    return tiled_point_clouds

result = tile_point_cloud(points, 10/4)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] != 0}

pc_tile_keys = np.array(list(result.keys()))

result.values()
result.keys()

pc_tile_pts = np.vstack(list(result.values()))
np.shape(pc_tile_pts)
for key in result:
    print(f"{key}: {len(result[key])} values")


def max_distance_to_plane_below(points, plane):
    """
    Calculate the maximum distance between a plane and a set of points below the plane in the z-direction.
    """

    def distance_to_plane(plane, point):
        """
        Calculate the distance between a plane and a point in the z-direction.
        """
        distance = np.abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3])
        if plane[2] > 0 and point[2] > -plane[3] / plane[2]:
            return 0
        return distance

    distances = [distance_to_plane(plane, point) for point in points if
                 plane[2] > 0 and point[2] <= -plane[3] / plane[2]]
    return max(distances)




for tile_name in result.keys():
    pc_array = result[tile_name]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_array)

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # inliers in red
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # outliers in grey

    max_distance = max_distance_to_plane_below(pc_array, plane_model)
    print(f"Maximum distance for tile {tile_name}: {max_distance}")

    df = pd.DataFrame(data=pc_array, columns=['x', 'y', 'z'])

    _df_to_las_conversion(df, address='Input', name=f'{tile_name}_output',
                          data_columns=['x', 'y', 'z'])