import pandas as pd
from manipulation_tools import _df_to_las_conversion
import open3d as o3d
import numpy as np
import os

print(os.getcwd())
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the point cloud
pcd = o3d.io.read_point_cloud("../Input/track2_pc_kmeans1.ply")

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)


def tile_point_cloud(points, tile_size):
    # Determine the minimum and maximum values for each dimension
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    # Compute the number of tiles along each dimension
    num_tiles_x = 3
    num_tiles_y = int(np.ceil((max_y - min_y) / tile_size))
    num_tiles_z = int(np.ceil((max_z - min_z) / tile_size))

    # Create a dictionary to store the tiled point clouds
    tiled_point_clouds = {}

    # Iterate over each tile
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            for k in range(num_tiles_z):
                # Compute the bounds of the tile
                x_min = min_x + i * (max_x - min_x) / num_tiles_x
                x_max = x_min + (max_x - min_x) / num_tiles_x
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


result = tile_point_cloud(points, 10 / 7)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 4000}
for key in result:
    print(f"{key}: {len(result[key])} values")


def avg_distance_to_plane_below(points, plane):
    """
    Calculate the average distance between a plane and the farthest 5 points below the plane in the z-direction.
    """

    def distance_to_plane(plane, point):
        """
        Calculate the distance between a plane and a point in the z-direction.
        """
        distance = np.abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3])
        return distance

    distances = [distance_to_plane(plane, point) for point in points]
    distances.sort(reverse=True)
    return sum(distances[:10]) / 10


def process_point_cloud(result):
    """
    Processes the point cloud data for each tile in the input result dictionary.

    Args:
        result (dict): A dictionary with tile names as keys and point cloud arrays as values.

    Returns:
        max_distances_cm (list): A list of the maximum distances of each tile from its plane, in centimeters.
    """
    max_distances = []
    tile_names = []
    tile_number = 1

    for tile_name in result.keys():
        pc_array = result[tile_name]
        if len(pc_array) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array)

        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

        max_distance = avg_distance_to_plane_below(pc_array, plane_model)
        print(f"Maximum distance for tile {tile_name} ({len(pc_array)} points): {max_distance}")

        df = pd.DataFrame(data=pc_array, columns=['x', 'y', 'z'])

        _df_to_las_conversion(df, address='../Input', name=f'{tile_name}_output',
                              data_columns=['x', 'y', 'z'])
        max_distances.append(max_distance)
        tile_names.append(tile_number)
        tile_number += 1

    max_distances_cm = [round(d * 100, 2) for d in max_distances]

    return max_distances_cm, tile_names


max_distances_cm, tile_names = process_point_cloud(result)

# Define colormap
norm = plt.Normalize(min(max_distances_cm), max(max_distances_cm))
cmap = cm.ScalarMappable(norm=norm, cmap=cm.Reds)

# Create bar chart with colored bars
fig, ax = plt.subplots()
bars = ax.bar(tile_names, max_distances_cm, width=0.5)

# Set color of bars
for i in range(len(bars)):
    bars[i].set_color(cmap.to_rgba(max_distances_cm[i]))

# Add colorbar legend
cbar = fig.colorbar(cmap)
cbar.set_label('Max Distance (cm)', fontsize=14)

# Add labels and title to chart
plt.xlabel('Tile', fontsize=14)
plt.ylabel('Max Distance (cm)', fontsize=14)
plt.title('Max Distance for Each Tile', fontsize=16)
plt.savefig('figure.png', dpi=300)
plt.show()

from pyecharts.charts import Bar
from pyecharts import options as opts


# Interactive Bar Chart
# data
tile_names = list(range(1, len(max_distances_cm) + 1))
# create the bar chart
bar = (
    Bar()
    .add_xaxis(tile_names)
    .add_yaxis("Max Distance (cm)", max_distances_cm, color="#1565C0")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Max Distance for Each Tile", subtitle="",
                                  title_textstyle_opts=opts.TextStyleOpts(font_weight='bold'),
                                  subtitle_textstyle_opts=opts.TextStyleOpts(font_weight='bold')),
        xaxis_opts=opts.AxisOpts(
            name='Tile',
            name_location='middle',
            name_gap=30,
            name_textstyle_opts=opts.TextStyleOpts(font_weight='bold', font_size=14),
            axislabel_opts=opts.LabelOpts(font_size=14),
        ),
        yaxis_opts=opts.AxisOpts(
            name='Max Distance (cm)',
            name_location='middle',
            name_gap=30,
            name_textstyle_opts=opts.TextStyleOpts(font_weight='bold', font_size=14),
            axislabel_opts=opts.LabelOpts(font_size=14),
        ),
        visualmap_opts=opts.VisualMapOpts(min_=min(max_distances_cm), max_=max(max_distances_cm),
                                          range_text=['High', 'Low'],
                                          range_color=['#d94e5d', '#eac736', '#50a3ba'][::-1]),
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),  # set is_show to False to remove data labels
        markpoint_opts=opts.MarkPointOpts(
            data=[opts.MarkPointItem(type_='max', name='Max Distance'),
                  opts.MarkPointItem(type_='min', name='Min Distance')],
            symbol_size=[60, 30],
            label_opts=opts.LabelOpts(font_size=14, position='inside', color='white')
        ),
        itemstyle_opts=opts.ItemStyleOpts(),
    )
)

# render the chart to a file
bar.render('max_distances.html')

# TODO: Check Seaborn heatmap. min max scale the max_distance_cm


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale

# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_cm)

# Apply min-max scaling to the max_distances array
max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_cm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(3)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 119  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)

# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='coolwarm', ax=ax)

# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} cm', f'{np.max(max_distances_arr):.2f} cm']) # add units here
# Set the title and axis labels
ax.set_title('Max Distance (cm) Heatmap')
ax.set_xlabel('Tile X-axis')
ax.set_ylabel('Tile Y-axis')

# Show the plot
plt.savefig('heatmap.png', dpi=300)

plt.show()
#####################convert the tiles with replaced z with max_distances to las files#################

from sklearn.preprocessing import minmax_scale

# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_cm)

# Apply min-max scaling to the max_distances array
max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
max_distances = list(max_distances_scaled)
new_results_scaled = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances[i]

def convert_tiles_las(new_results):

    for tile_name in new_results.keys():
        pc_array_new = new_results[tile_name]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array_new)

        df = pd.DataFrame(data=pc_array_new, columns=['x', 'y', 'z'])

        _df_to_las_conversion(df, address='../Input', name=f'{tile_name}_output_maxdist',
                              data_columns=['x', 'y', 'z'])


convert_tiles_las(new_results)


