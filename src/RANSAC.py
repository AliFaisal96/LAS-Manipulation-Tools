import pickle
import numpy as np
import open3d as o3d

# load the output from the file
with open("PtCld_Tiles.pkl", "rb") as f:
    Ptcld_Tiles = pickle.load(f)


# Define a function to detect planes from a point cloud using Open3D RANSAC
def detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations):
    # Create an empty list to store the plane models
    planes = []
    # Create a copy of the point cloud
    remaining_points = point_cloud.copy()
    # Loop until no more planes are found or the point cloud is empty
    while True:
        # Segment a plane from the point cloud using Open3D RANSAC
        plane_model, inliers = remaining_points.segment_plane(distance_threshold, ransac_n, num_iterations)
        # If no inliers are found, break the loop
        if len(inliers) == 0:
            break
        # Extract the inlier points as a new point cloud
        inlier_cloud = remaining_points.select_by_index(inliers)
        # Append the plane model and the inlier cloud to the planes list
        planes.append((plane_model, inlier_cloud))
        # Remove the inlier points from the remaining point cloud
        remaining_points = remaining_points.select_by_index(inliers, invert=True)
    # Return the planes list
    return planes


# Define a dictionary that contains keys for point cloud tiles and arrays of the point cloud data
point_cloud_dict = {
    # Example data, you can replace with your own
    "tile_1": [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
    "tile_2": [[2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0], [2, 0, 1], [2, 1, 1], [3, 0, 1], [3, 1, 1]],
    "tile_3": [[0, 2, 0], [0, 3, 0], [1, 2, 0], [1, 3, 0], [0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1]],
    "tile_4": [[2, 2, 0], [2, 3, 0], [3, 2, 0], [3, 3, 0], [2, 2, 1], [2, 3, 1], [3, 2, 1], [3, 3, 1]]
}

# Define the parameters for RANSAC
distance_threshold = 0.01  # Max distance a point can be from the plane model, and still be considered an inlier
ransac_n = 3  # Number of points to sample as a minimum data set
num_iterations = 100  # Number of iterations to run

# Loop through the keys of the dictionary and apply the detect_planes function to each array of point cloud data
for key, value in point_cloud_dict.items():
    # Convert the array of point cloud data to an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(value)
    # Detect the planes from the point cloud using Open3D RANSAC
    planes = detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations)
    # Print the results
    print(f"Planes detected for {key}:")
    for i, (plane_model, inlier_cloud) in enumerate(planes):
        print(f"Plane {i + 1}:")
        print(f"Plane model: {plane_model}")
        print(f"Number of inlier points: {len(inlier_cloud.points)}")
        print(f"Inlier points: {inlier_cloud.points}")
    print()

# faced an error with the copy method using open3d
# %%
# Import Open3D library
import open3d as o3d


# Define a function to detect planes from a point cloud using Open3D RANSAC
def detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations):
    # Create an empty list to store the plane models
    planes = []
    # Create a numpy array from the point cloud points
    point_cloud_array = np.asarray(point_cloud.points)
    # Loop until no more planes are found or the point cloud is empty
    while True:
        # Segment a plane from the point cloud using Open3D RANSAC
        plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
        # If no inliers are found, break the loop
        if len(inliers) == 0:
            break
        # Extract the inlier points as a new point cloud
        inlier_cloud = point_cloud.select_by_index(inliers)
        # Append the plane model and the inlier cloud to the planes list
        planes.append((plane_model, inlier_cloud))
        # Remove the inlier points from the point cloud array
        point_cloud_array = np.delete(point_cloud_array, inliers, axis=0)
        # Create a new point cloud object from the remaining points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_array)
    # Return the planes list
    return planes


# Define a dictionary that contains keys for point cloud tiles and arrays of the point cloud data
point_cloud_dict = {
    # Example data, you can replace with your own
    "tile_1": [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
    "tile_2": [[2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0], [2, 0, 1], [2, 1, 1], [3, 0, 1], [3, 1, 1]],
    "tile_3": [[0, 2, 0], [0, 3, 0], [1, 2, 0], [1, 3, 0], [0, 2, 1], [0, 3, 1], [1, 2, 1], [1, 3, 1]],
    "tile_4": [[2, 2, 0], [2, 3, 0], [3, 2, 0], [3, 3, 0], [2, 2, 1], [2, 3, 1], [3, 2, 1], [3, 3, 1]]
}

# Define the parameters for RANSAC
distance_threshold = 0.01  # Max distance a point can be from the plane model, and still be considered an inlier
ransac_n = 3  # Number of points to sample as a minimum data set
num_iterations = 100  # Number of iterations to run

# Loop through the keys of the dictionary and apply the detect_planes function to each array of point cloud data
for key, value in Ptcld_Tiles.items():
    # Convert the array of point cloud data to an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(value)
    # Detect the planes from the point cloud using Open3D RANSAC
    planes = detect_planes(point_cloud, distance_threshold, ransac_n, num_iterations)
    # Print the results
    print(f"Planes detected for {key}:")
    for i, (plane_model, inlier_cloud) in enumerate(planes):
        print(f"Plane {i + 1}:")
        print(f"Plane model: {plane_model}")
        print(f"Number of inlier points: {len(inlier_cloud.points)}")
        print(f"Inlier points: {inlier_cloud.points}")
    print()

pcd = o3d.io.read_point_cloud("Input/track1_filtered.ply")

planes = detect_planes(pcd, distance_threshold, ransac_n, num_iterations)

# error with ransac sample points
# %%

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA

# set RANSAC parameters
ransac = RANSACRegressor(
    base_estimator=PCA(n_components=3),
    residual_threshold=0.1,
    max_trials=1000,
    stop_probability=0.999,
    random_state=0
)


# define function to fit plane with RANSAC on point cloud
def fit_plane_with_ransac(pc_tile):
    # fit PCA to point cloud
    pca = PCA(n_components=3)
    X = pca.fit_transform(pc_tile)

    # apply RANSAC to find inliers for plane model
    y = np.zeros(X.shape[0])  # dummy y input with same number of rows as X

    ransac = RANSACRegressor(
        base_estimator=None,
        residual_threshold=0.1,
        max_trials=1000,
        stop_probability=0.999,
        random_state=0
    )
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_

    # extract inlier points
    inliers = pc_tile[inlier_mask, :]

    # extract plane parameters from PCA
    normal = pca.components_[2, :]
    point = np.mean(inliers, axis=0)

    return (point, normal, inlier_mask)


# define function to apply RANSAC to each tile in dictionary
def apply_ransac_to_dict(pc_dict):
    # initialize empty dictionary for inliers and planes
    inlier_dict = {}
    plane_dict = {}

    # iterate over point cloud tiles
    for tile_name, pc_tile in pc_dict.items():
        # check if tile has any points
        if pc_tile.shape[0] > 0:
            # fit plane with RANSAC
            point, normal, inlier_mask = fit_plane_with_ransac(pc_tile)

            # add inliers and plane parameters to dictionaries
            inlier_dict[tile_name] = inlier_mask
            plane_dict[tile_name] = (point, normal)

    return (inlier_dict, plane_dict)


apply_ransac_to_dict(Ptcld_Tiles)

# error with y argument
# %% applying ransac to an individual tile and extracting the inliers, outliers and plane equation, along with
#the difference in the z values between the inliers and outliers (max z in inliers - min z in outliers)

pcd = o3d.geometry.PointCloud()
pc_array1 = Ptcld_Tiles[list(Ptcld_Tiles.keys())[0]]
pcd.points = o3d.utility.Vector3dVector(pc_array1)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
# now I have the plane equation in plane model, and the points inside within the plane in inliers
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1.0, 0, 0]) #inliers in red
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6]) #outliers in grey
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.spatial.distance import cdist

# this doesn't take into account the z axis
inlier_z = np.asarray(inlier_cloud.points)[:, 2]
outlier_z = np.asarray(outlier_cloud.points)[:, 2]
# distances = cdist(np.asarray(inlier_cloud.points), np.asarray(outlier_cloud.points))
# max_distance = np.max(distances)
np.average(inlier_z) - np.average(outlier_z) #x10 for cm, x100 mm: 13 mm difference
