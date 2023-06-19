import pandas as pd
from manipulation_tools import _df_to_las_conversion
import open3d as o3d
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import matplotlib as mpl


print(os.getcwd())


# Load the point cloud
pcd = o3d.io.read_point_cloud("Input/Track 1/track1_pc_cleaned.ply")

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)


def tile_point_cloud(points, tile_size, num_tiles_x):
    # Determine the minimum and maximum values for each dimension
    min_x, min_y, min_z = np.min(points, axis=0)
    max_x, max_y, max_z = np.max(points, axis=0)

    # Compute the number of tiles along each dimension
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


def save_all_tiles_to_csv(tiled_point_clouds, output_file):
    all_points = []
    for tile_key, tile_points in tiled_point_clouds.items():
        tile_df = pd.DataFrame(tile_points, columns=["x", "y", "z"])
        tile_df["tile"] = str(tile_key)
        all_points.append(tile_df)

    combined_df = pd.concat(all_points, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

result = tile_point_cloud(points, 10 / 6.5, 2)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 4000}
for key in result:
    print(f"{key}: {len(result[key])} values")


output_file = "Input/Track 1/tiled_point_clouds.csv"
save_all_tiles_to_csv(result, output_file)


# def plot_tiles_3d(tiled_point_clouds, selected_tile_keys):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     for tile_key in selected_tile_keys:
#         if tile_key in tiled_point_clouds:
#             tile_points = tiled_point_clouds[tile_key]
#             ax.scatter(tile_points[:, 0], tile_points[:, 1], tile_points[:, 2], label=str(tile_key))
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     plt.show()
#
#
# # Select the tile keys you want to plot (use the keys from your tiled point cloud)
# selected_tile_keys = [(0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 0)]
#
# # Plot the selected tiles
# plot_tiles_3d(result, selected_tile_keys)
def avg_distance_to_plane_below(points, plane):
    """
    Calculate the average distance between a plane and the farthest 20 points below the plane in the z-direction.
    """

    def distance_to_plane(plane, point):
        """
        Calculate the distance between a plane and a point in the z-direction.
        """
        distance = np.abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3])
        return distance

    distances = [distance_to_plane(plane, point) for point in points]
    distances.sort(reverse=True)
    return sum(distances[:50]) / 50


def process_point_cloud(result, output_dir):
    """
    Processes the point cloud data for each tile in the input result dictionary.

    Args:
        result (dict): A dictionary with tile names as keys and point cloud arrays as values.

    Returns:
        max_distances_mm (list): A list of the maximum distances of each tile from its plane, in centimeters.
        :param output_dir: location of file
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

        _df_to_las_conversion(df, address=output_dir, name=f'{tile_name}_output',
                              data_columns=['x', 'y', 'z'])

        max_distances.append(max_distance)
        tile_names.append(tile_number)
        tile_number += 1

    max_distances_mm = [round(d * 1000, 2) for d in max_distances]
    output_df = pd.DataFrame({'Tile Name': tile_names, 'Max Distance (mm)': max_distances_mm})

    # Export the DataFrame to a CSV file
    output_df.to_csv(os.path.join(output_dir, 'distances_and_tile_names.csv'), index=False)

    return max_distances_mm, tile_names

output_dir = "Input/Track 1"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)

# Define colormap
norm = plt.Normalize(min(max_distances_mm), max(max_distances_mm))
cmap = cm.ScalarMappable(norm=norm, cmap=cm.Reds)

# Create bar chart with colored bars
fig, ax = plt.subplots()
bars = ax.bar(tile_names, max_distances_mm, width=0.5)

# Set color of bars
for i in range(len(bars)):
    bars[i].set_color(cmap.to_rgba(max_distances_mm[i]))

# Add colorbar legend
cbar = fig.colorbar(cmap)
cbar.set_label('Max Distance (mm)', fontsize=14)

# Add labels and title to chart
plt.xlabel('Tile', fontsize=14)
plt.ylabel('Max Distance (mm)', fontsize=14)
plt.title('Max Distance for Each Tile', fontsize=16)
plt.savefig('figure.png', dpi=300)
plt.show()

# from pyecharts.charts import Bar
# from pyecharts import options as opts


# # Interactive Bar Chart ####doing it remotely on vscode, seems environment issue####
# # data
# tile_names = list(range(1, len(max_distances_mm) + 1))
# # create the bar chart
# bar = (
#     Bar()
#     .add_xaxis(tile_names)
#     .add_yaxis("Max Distance (cm)", max_distances_mm, color="#1565C0")
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="Max Distance for Each Tile", subtitle="",
#                                   title_textstyle_opts=opts.TextStyleOpts(font_weight='bold'),
#                                   subtitle_textstyle_opts=opts.TextStyleOpts(font_weight='bold')),
#         xaxis_opts=opts.AxisOpts(
#             name='Tile',
#             name_location='middle',
#             name_gap=30,
#             name_textstyle_opts=opts.TextStyleOpts(font_weight='bold', font_size=14),
#             axislabel_opts=opts.LabelOpts(font_size=14),
#         ),
#         yaxis_opts=opts.AxisOpts(
#             name='Max Distance (cm)',
#             name_location='middle',
#             name_gap=30,
#             name_textstyle_opts=opts.TextStyleOpts(font_weight='bold', font_size=14),
#             axislabel_opts=opts.LabelOpts(font_size=14),
#         ),
#         visualmap_opts=opts.VisualMapOpts(min_=min(max_distances_mm), max_=max(max_distances_mm),
#                                           range_text=['High', 'Low'],
#                                           range_color=['#d94e5d', '#eac736', '#50a3ba'][::-1]),
#     )
#     .set_series_opts(
#         label_opts=opts.LabelOpts(is_show=False),  # set is_show to False to remove data labels
#         markpoint_opts=opts.MarkPointOpts(
#             data=[opts.MarkPointItem(type_='max', name='Max Distance'),
#                   opts.MarkPointItem(type_='min', name='Min Distance')],
#             symbol_size=[60, 30],
#             label_opts=opts.LabelOpts(font_size=14, position='inside', color='white')
#         ),
#         itemstyle_opts=opts.ItemStyleOpts(),
#     )
# )
#
# # render the chart to a file
# bar.render('max_distances.html')



# TODO: Check Seaborn heatmap. min max scale the max_distance_mm


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale

# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(2)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 113  # Calculate the index of the current tile
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
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here

# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left Lane', 'Right Lane']  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

# Update the y-axis tick labels
y_tick_labels = [0, 13, 26, 39, 52, 65, 77, 90, 103, 116, 129, 142, 155, 168, 181]

# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)
# Show the plot
plt.savefig('heatmap_track1.svg', format="svg")

plt.show()
#####################convert the tiles with replaced z with max_distances to las files#################

from sklearn.preprocessing import minmax_scale

# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

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

        _df_to_las_conversion(df, address='../Input/Track 1', name=f'{tile_name}_output_maxdist',
                              data_columns=['x', 'y', 'z'])


convert_tiles_las(new_results)

#%% Track 1
pcd = o3d.io.read_point_cloud("Input/Hwy 32/Track 1/track1_pc_cleaned.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 10 / 6.5, 2)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 4000}
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/Hwy 32/Track 1/outputs"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)


# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(2)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 113  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)
plt.rcParams['font.family'] = 'Times New Roman'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='plasma', ax=ax)

# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left Lane',"Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

# Update the y-axis tick labels
y_tick_labels = [0, 13, 26, 39, 52, 65, 77, 90, 103, 116, 129, 142, 155, 168, 181]
# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)
# Show the plot
plt.savefig('Input/Hwy 32/Track 1/outputs/heatmap_track1.svg', format="svg")
plt.savefig('Input/Hwy 32/Track 1/outputs/heatmap_track1.png', format="png")
plt.show()


#%%track 2 right turn####
pcd = o3d.io.read_point_cloud("Input/Hwy 32/Track 2 R/track2_rightturn_cleaned.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 10 / 7, 3)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 4000}
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/Hwy 32/Track 2 R/outputs"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)


# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(3)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 65  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)

# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='plasma', ax=ax)
plt.rcParams['font.family'] = 'Times New Roman'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# Create the heatmap using Seaborn
# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left Lane', 'Middle Lane', "Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

# Update the y-axis tick labels
y_tick_labels = [0, 7, 14, 21, 28, 34, 41, 48, 55, 62, 69, 76, 83, 89, 96]
# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)
# Show the plot
plt.savefig('Input/Hwy 32/Track 2 R/outputs/heatmap_rightturn_track2.svg', format="svg")
plt.savefig('Input/Hwy 32/Track 2 R/outputs/heatmap_rightturn_track2.png', format="png")
plt.show()
#%%track 2 left turn####
pcd = o3d.io.read_point_cloud("Input/Hwy 32/Track 2 L/track2_leftturn_cleaned.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 10 / 9, 3)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 1000}
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/Hwy 32/Track 2 L/outputs"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)

# # Define colormap
# norm = plt.Normalize(min(max_distances_mm), max(max_distances_mm))
# cmap = cm.ScalarMappable(norm=norm, cmap=cm.Reds)
#
# # Create bar chart with colored bars
# fig, ax = plt.subplots()
# bars = ax.bar(tile_names, max_distances_mm, width=0.5)
#
# # Set color of bars
# for i in range(len(bars)):
#     bars[i].set_color(cmap.to_rgba(max_distances_mm[i]))
#
# # Add colorbar legend
# cbar = fig.colorbar(cmap)
# cbar.set_label('Max Distance (mm)', fontsize=14)
#
# # Add labels and title to chart
# plt.xlabel('Tile', fontsize=14)
# plt.ylabel('Max Distance (mm)', fontsize=14)
# plt.title('Max Distance for Each Tile', fontsize=16)
# plt.savefig('figure.png', dpi=300)
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import minmax_scale

# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(3)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 56  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)

# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='plasma', ax=ax)

plt.rcParams['font.family'] = 'Times New Roman'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left-turn Lane', 'Middle Lane', "Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

# Update the y-axis tick labels
y_tick_labels = [ 0, 4, 9, 13, 18, 22, 27, 31, 35, 40, 44, 49, 53, 58, 62]

# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)
# Show the plot
plt.savefig('Input/Hwy 32/Track 2 L/outputs/heatmap_left_track2.svg', format="svg")
plt.savefig('Input/Hwy 32/Track 2 L/outputs/heatmap_left_track2.png', format="png")


plt.show()


#%%BCMoTI Section 2 Classified
import numpy as np
import open3d as o3d
import pandas as pd
from manipulation_tools import _df_to_las_conversion
pcd = o3d.io.read_point_cloud("Input/BCMoTI/Section 2/Tile_10-2.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 10, 2)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 1000}
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/BCMoTI/Section 2"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)

#%%Alberta Dataset L1L2
pcd = o3d.io.read_point_cloud("Input/Alberta/Road_L1L2.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 10, 2)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 1000}
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/Alberta"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)
# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
# max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
# max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(2)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 245  # Calculate the index of the current tile
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
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left-turn Lane', "Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

# Update the y-axis tick labels
# y_tick_labels = [ 0, 4, 9, 13, 18, 22, 27, 31, 35, 40, 44, 49, 53, 58, 62]
#
# # Set the number of y-axis ticks to match the number of tick labels
# num_y_ticks = len(y_tick_labels)
# ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
# ax.set_yticklabels(y_tick_labels)
# # Show the plot
# plt.savefig('heatmap_left_track2.svg', format="svg")

plt.show()


#%%Miette Road
pcd = o3d.io.read_point_cloud("Input/Miette Road/miette_road.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 1, 2)

# remove empty tiles (arrays shape 0)
result = {k: v for k, v in result.items() if v.shape[0] >= 1000}
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/Miette Road/outputs"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)
# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
# max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
# max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(2)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 130  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)

# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='plasma', ax=ax)

plt.rcParams['font.family'] = 'Times New Roman'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left Lane', "Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

#Update the y-axis tick labels
road_length = 130  # Length of the road in meters
num_ticks = 15  # Number of tick labels

interval = road_length / (num_ticks - 1)  # Calculate the interval between each tick

# Generate the tick labels based on the interval
y_tick_labels = [int(i * interval) for i in range(num_ticks)]

# Add the unit of measurement to each tick label
y_tick_labels = [f'{label} m' for label in y_tick_labels]

# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)

# Show the plot
plt.savefig('Input/Miette Road/outputs/Miette.svg', format="svg")
plt.savefig('Input/Miette Road/outputs/Miette.png', format="png")
plt.show()

#%%TWP510_3lane Road
pcd = o3d.io.read_point_cloud("Input/TWP510 Road/TWP510_3lane/TWP510_3lane.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 1, 3)

# remove empty tiles (arrays shape 0)
#result = {k: v for k, v in result.items() if v.shape[0] >= 1000} #not needed here since the point density is low
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/TWP510 Road/TWP510_3lane/outputs"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)
# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
# max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
# max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(3)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 96  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)

# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='plasma', ax=ax)

plt.rcParams['font.family'] = 'Times New Roman'
new_rc_params = {'text.usetex': False, "svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left Lane','Middle Lane', "Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

#Update the y-axis tick labels
road_length = 96  # Length of the road in meters
num_ticks = 15  # Number of tick labels

interval = road_length / (num_ticks - 1)  # Calculate the interval between each tick

# Generate the tick labels based on the interval
y_tick_labels = [int(i * interval) for i in range(num_ticks)]

# Add the unit of measurement to each tick label
y_tick_labels = [f'{label} m' for label in y_tick_labels]

# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)

# Show the plot
plt.savefig('Input/TWP510 Road/TWP510_3lane/outputs/TWP510_3lane.svg', format="svg")
plt.savefig('Input/TWP510 Road/TWP510_3lane/outputs/TWP510_3lane.png', format="png")
plt.show()

#%%TWP510_2lane Road
pcd = o3d.io.read_point_cloud("Input/TWP510 Road/TWP510_2lane/TWP510_2lane.ply")
points = np.asarray(pcd.points)
result = tile_point_cloud(points, 1, 2)

# remove empty tiles (arrays shape 0)
#result = {k: v for k, v in result.items() if v.shape[0] >= 1000} #not needed here since the point density is low
for key in result:
    print(f"{key}: {len(result[key])} values")

output_dir = "Input/TWP510 Road/TWP510_2lane/outputs"
max_distances_mm, tile_names = process_point_cloud(result, output_dir)
# Convert the max_distances list to a numpy array
max_distances_arr = np.array(max_distances_mm)

# Apply min-max scaling to the max_distances array
# max_distances_scaled = minmax_scale(max_distances_arr, feature_range=(0, 100))

# Replace the original max_distances list with the scaled values
# max_distances = list(max_distances_scaled)
new_results = result.copy()  # get a new result but replace the z values with the max distances
for i, tile in enumerate(new_results.keys()):
    new_results[tile][:, 2] = max_distances_mm[i]

###create heatmap####
# Create a 2D array of the max distances
max_distances = []

for tile in new_results.values():
    max_distances.append(tile[0][2])

# Create a list of lists to hold the distances for each tile
distances_by_tile = [[] for _ in range(2)]

# Iterate over the distances and append them to the appropriate inner list
for i, distance in enumerate(max_distances):
    tile_index = i // 95  # Calculate the index of the current tile
    distances_by_tile[tile_index].append(distance)

# Stack the lists of distances horizontally using np.column_stack
max_distances_arr = np.column_stack(distances_by_tile)

# Create the heatmap using Seaborn
fig, ax = plt.subplots()
sns.heatmap(max_distances_arr, cmap='plasma', ax=ax)

# Inverse the y-axis
ax.invert_yaxis()

# Add a colorbar
cbar = ax.collections[0].colorbar
cbar.set_ticks([np.min(max_distances_arr), np.max(max_distances_arr)])
cbar.set_ticklabels([f'{np.min(max_distances_arr):.2f} mm', f'{np.max(max_distances_arr):.2f} mm']) # add units here
# Set the title and axis labels
ax.set_title('Mean Distance (mm) Heatmap')
# ax.set_xlabel('Lane Tile')
ax.set_ylabel('Longitudinal Distance (m)')
# Update the x-axis tick labels
x_tick_labels = ['Left Lane', "Right Lane"]  # Update these labels as needed
ax.set_xticklabels(x_tick_labels)

#Update the y-axis tick labels
road_length = 95  # Length of the road in meters
num_ticks = 15  # Number of tick labels

interval = road_length / (num_ticks - 1)  # Calculate the interval between each tick

# Generate the tick labels based on the interval
y_tick_labels = [int(i * interval) for i in range(num_ticks)]

# Add the unit of measurement to each tick label
y_tick_labels = [f'{label} m' for label in y_tick_labels]

# Set the number of y-axis ticks to match the number of tick labels
num_y_ticks = len(y_tick_labels)
ax.set_yticks(np.linspace(0, len(max_distances_arr) - 1, num_y_ticks))
ax.set_yticklabels(y_tick_labels)

# Show the plot
plt.savefig('Input/TWP510 Road/TWP510_2lane/outputs/TWP510_2lane.svg', format="svg")
plt.savefig('Input/TWP510 Road/TWP510_2lane/outputs/TWP510_2lane.png', format="png")
plt.show()


#looks nice but not better than the seaborn heatmap
# from plotnine import *
# from plotnine.data import mpg
#
# # Create a 2D array of the max distances
# max_distances = []
#
# for tile in new_results.values():
#     max_distances.append(tile[0][2])
#
# # Create a list of lists to hold the distances for each tile
# distances_by_tile = [[] for _ in range(2)]
#
# # Iterate over the distances and append them to the appropriate inner list
# for i, distance in enumerate(max_distances):
#     tile_index = i // 95  # Calculate the index of the current tile
#     distances_by_tile[tile_index].append(distance)
#
# # Stack the lists of distances horizontally using np.column_stack
# max_distances_arr = np.column_stack(distances_by_tile)
#
# # Create the dataframe for the heatmap
# df = pd.DataFrame(max_distances_arr, columns=['Left Lane', 'Right Lane'])
# df['Longitudinal Distance'] = range(0, len(df))
#
# # Reshape the dataframe for plotting
# df = df.melt('Longitudinal Distance', var_name='Lane', value_name='Distance')
#
# # Create the heatmap using plotnine
# heatmap = (
#     ggplot(df, aes(x='Lane', y='Longitudinal Distance', fill='Distance'))
#     + geom_tile(color='white')
#     + scale_fill_cmap(cmap_name='viridis', name='Mean Distance (mm)')
#     + labs(title='Mean Distance (mm) Heatmap', x='', y='Longitudinal Distance (m)')
#     + theme_minimal()
#     + theme(axis_text_x=element_text(angle=90, hjust=1)))
#
# # Save the plot
# heatmap.save('Input/TWP510 Road/TWP510_2lane/outputs/TWP510_2lane.png', dpi=300)
#
# # Show the plot
# print(heatmap)

d