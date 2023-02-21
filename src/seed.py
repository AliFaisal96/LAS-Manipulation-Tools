import numpy as np

from src.LAS_tools import df_elevation2

df_elevation2

def region_growing(points, seed_index, distance_threshold):
    """
    Filters a point cloud to keep only the road segment using a region growing algorithm.

    Parameters:
        - points: Numpy array of shape (n, m) containing the point cloud data, where n is the number of points and m is the number of features.
        - seed_index: Integer representing the index of the seed point in the points array.
        - distance_threshold: Float representing the maximum distance from a road point for a point to be included in the road segment.

    Returns:
        - road_points: Numpy array of shape (k, m) containing the filtered road segment, where k is the number of points in the road segment.
    """
    # Initialize list to store the road points
    road_points = []

    # Add the seed point to the list of road points
    road_points.append(points[seed_index])

    # Loop through the remaining points
    for i in range(points.shape[0]):
        if i == seed_index:
            continue
        # Calculate the distance from the current point to each road point
        distances = np.linalg.norm(points[i] - road_points, axis=1)
        # If the minimum distance is less than the distance threshold, add the current point to the road points list
        if np.min(distances) < distance_threshold:
            road_points.append(points[i])

    # Convert the list of road points to a Numpy array
    road_points = np.array(road_points)

    return road_points

# Load the point cloud data into a Numpy array

points = df_elevation2.to_numpy()
points = points[: , 1:4]
points.shape[1]
# Choose a seed point
seed_index = 0

# Define the distance threshold
distance_threshold = 0.1

# Filter the point cloud to keep only the road segment
road_points = region_growing(points, seed_index, distance_threshold)

# Visualize the filtered road segment
plt.scatter(road_points[:, 0], road_points[:, 1], c='red')
plt.scatter(points[:, 0], points[:, 1], c='blue')
plt.show()

x = 3
x

#%%

def region_growing(points, seed_index, distance_threshold):
    """
    Filters a point cloud to keep only the road segment using a region growing algorithm.

    Parameters:
        - points: Numpy array of shape (n, m) containing the point cloud data, where n is the number of points and m is the number of features.
        - seed_index: Integer representing the index of the seed point in the points array.
        - distance_threshold: Float representing the maximum distance from a road point for a point to be included in the road segment.

    Returns:
        - road_points: Numpy array of shape (k, m) containing the filtered road segment, where k is the number of points in the road segment.
    """
    # Initialize list to store the indices of the road points
    road_indices = [seed_index]

    # Initialize a Boolean array to keep track of which points have been processed
    processed = np.zeros(points.shape[0], dtype=bool)
    processed[seed_index] = True

    # Loop until all points have been processed
    while not np.all(processed):
        # Initialize a list to store the indices of the newly added road points
        new_road_indices = []

        # Loop through the indices of the current road points
        for i in road_indices:
            # Calculate the distances from the current road point to all the other points
            distances = np.linalg.norm(points - points[i], axis=1)

            # Find the indices of the points that are within the distance threshold and have not been processed
            new_indices = np.where((distances < distance_threshold) & (~processed))[0]

            # Add the new indices to the list of newly added road points
            new_road_indices.extend(new_indices)

            # Mark the newly added points as processed
            processed[new_indices] = True

        # Update the list of road indices with the newly added road points
        road_indices = new_road_indices

    # Extract the filtered road segment from the point cloud data
    road_points = points[road_indices]

    return road_points

# Load the point cloud data into a Numpy array
points = df_elevation2.to_numpy()

# Choose a seed point
seed_index = 0

# Define the distance threshold
distance_threshold = 0.1

# Filter the point cloud to keep only the road segment
road_points = region_growing(points, seed_index, distance_threshold)

# Visualize the filtered road segment
plt.scatter(road_points[:, 0], road_points[:, 1], c='red')
plt.scatter(points[:, 0], points[:, 1], c='blue')
plt.show()
