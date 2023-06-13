import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

new_directory = 'C:/Users/unbre\OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master'
os.chdir(new_directory)
print(os.getcwd())


def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points


def array_to_dataframe(point_cloud_array):
    data = pd.DataFrame(point_cloud_array, columns=["X", "Y", "Z"])
    return data


def interpolate_profiles(data, spacing, num, wheel_paths=True):
    if wheel_paths:
        y_min, y_max = data["Y"].min(), data["Y"].max()
        profiles = []

        for x in np.arange(data["X"].min(), data["X"].max(), spacing):
            profile_data = data[(data["X"] >= x) & (data["X"] < x + spacing)].sort_values(by="Y")
            unique_y_count = len(profile_data["Y"].unique())

            if not profile_data.empty and unique_y_count >= 2:
                y = profile_data["Y"].values
                z = profile_data["Z"].values
                y_new = np.linspace(y_min, y_max, num=num)
                z_new = np.interp(y_new, y, z)
                profile = np.column_stack((y_new, z_new))
                profiles.append(profile)

        return profiles
    else:
        raise NotImplementedError("Centerline profile extraction not implemented")


def clean_data(data):
    cleaned_data = data.drop_duplicates(subset=["X", "Y"])
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
    cleaned_data = cleaned_data.dropna()

    return cleaned_data


def filter_profiles(profiles, indices_to_remove, remove_last_point=True):
    filtered_profiles = [profile for i, profile in enumerate(profiles) if i not in indices_to_remove]

    if remove_last_point:
        for i, profile in enumerate(filtered_profiles):
            filtered_profiles[i] = filtered_profiles[i][:-1, :]

    return filtered_profiles


def normalize_profiles(profiles):
    normalized_profiles = []
    for profile in profiles:
        y_min = profile[:, 0].min()
        z_min = profile[:, 1].min()
        y_normalized = profile[:, 0] - y_min
        z_normalized = profile[:, 1] - z_min
        normalized_profile = np.column_stack((y_normalized, z_normalized))
        normalized_profiles.append(normalized_profile)
    return normalized_profiles


def save_profiles(profiles):
    for i, profile in enumerate(profiles):
        filename = f"Track 1_Profile{i + 1}.txt"
        np.savetxt(filename, profile)


def visualize_profiles(profiles, file_path):
    plt.figure(figsize=(10, 6))

    for i, profile in enumerate(profiles):
        plt.plot(profile[:, 0], profile[:, 1], label=f'Profile {i + 1}')

    plt.xlabel('Longitudinal Length (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Longitudinal Profiles')
    plt.legend()

    # Save the figure using the provided file path
    svg_file_path = file_path + '.svg'
    png_file_path = file_path + '.png'

    plt.savefig(svg_file_path, format="svg")
    plt.savefig(png_file_path, format="png")

    plt.show()


file_path = "Input/TWP510 Road/TWP510_3lane/TWP510_3lane.ply"
output_dir = "Input/TWP510 Road/TWP510_3lane/outputs"
profiles_to_remove = []
spacing = 1.2  # to select the number of profiles. It is the spacing taken between the profiles along the road width


# filter_length = 0.25  # Moving average filter length in meters
# num_points_per_profile = 100  # Number of points in each profile
# profile_length = 130
# window_size = int(filter_length / (profile_length / num_points_per_profile))

def main(file_path, output_dir, profiles_to_remove, spacing, window_size, remove_last_point=True):
    # Load point cloud data
    point_cloud_array = load_point_cloud(file_path)
    point_cloud_data = array_to_dataframe(point_cloud_array)
    point_cloud_data = clean_data(point_cloud_data)

    # Interpolate profiles along wheel paths
    profiles = interpolate_profiles(point_cloud_data, spacing, num=384)

    # Filter out specific profiles
    filtered_profiles = filter_profiles(profiles, profiles_to_remove, remove_last_point=True)

    # Normalize profiles
    normalized_profiles = normalize_profiles(filtered_profiles)

    # Remove the last two points from each profile
    normalized_profiles = [profile[:-2] for profile in normalized_profiles]

    # Visualize the smoothed profiles
    visualize_profiles(normalized_profiles, 'Input/TWP510 Road/TWP510_3lane/outputs/TWP510_3lane_Profiles')

    # Save the smoothed profiles to text files
    for i, profile in enumerate(normalized_profiles):
        filename = f"{output_dir}/Profile{i + 1}.txt"
        np.savetxt(filename, profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point cloud data and generate longitudinal profiles.")
    parser.add_argument("--input-file", required=True, help="Path to the input point cloud file")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    parser.add_argument("--remove", nargs="+", type=int, default=[0, 11], help="Indices of profiles to remove")
    parser.add_argument("--spacing", type=int, default=1, help="Spacing between profile sections")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for the moving average filter")
    parser.add_argument("--remove_last_point", action="store_true", default=False,
                        help="Remove the last point in each profile")
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.remove, args.spacing, args.window_size, args.remove_last_point)

df = pd.read_csv('output_figures/Profiles/TWP510 3 Lane/IRI Results/output10.txt', sep='\s+')

# %% Miette Profiles

file_path = "Input/Miette Road/miette_road.ply"
output_dir = "Input/Miette Road/outputs"
profiles_to_remove = []
spacing = 1  # to select the number of profiles. It is the spacing taken between the profiles along the road width


# filter_length = 0.25  # Moving average filter length in meters
# num_points_per_profile = 100  # Number of points in each profile
# profile_length = 130
# window_size = int(filter_length / (profile_length / num_points_per_profile))

def main(file_path, output_dir, profiles_to_remove, spacing, window_size, remove_last_point=True):
    # Load point cloud data
    point_cloud_array = load_point_cloud(file_path)
    point_cloud_data = array_to_dataframe(point_cloud_array)
    point_cloud_data = clean_data(point_cloud_data)

    # Interpolate profiles along wheel paths
    profiles = interpolate_profiles(point_cloud_data, spacing, num=520)

    # Filter out specific profiles
    filtered_profiles = filter_profiles(profiles, profiles_to_remove, remove_last_point=True)

    # Normalize profiles
    normalized_profiles = normalize_profiles(filtered_profiles)

    # Remove the last two points from each profile
    normalized_profiles = [profile[:-2] for profile in normalized_profiles]

    # Visualize the smoothed profiles
    visualize_profiles(normalized_profiles, "Input/Miette Road/outputs/Miette_Profiles")

    # Save the smoothed profiles to text files
    for i, profile in enumerate(normalized_profiles):
        filename = f"{output_dir}/Profile{i + 1}.txt"
        np.savetxt(filename, profile)


df = pd.read_csv('output_figures/Profiles/Miette_Road/IRI Results/output6.txt', sep='\s+')
#%% Hwy 32 Track 1
file_path = "Input/Hwy 32/Track 1/track1_pc_cleaned.ply"
output_dir = "Input/Hwy 32/Track 1/outputs"
profiles_to_remove = [0, 11]
spacing = 1  # to select the number of profiles. It is the spacing taken between the profiles along the road width


# filter_length = 0.25  # Moving average filter length in meters
# num_points_per_profile = 100  # Number of points in each profile
# profile_length = 130
# window_size = int(filter_length / (profile_length / num_points_per_profile))

def main(file_path, output_dir, profiles_to_remove, spacing, window_size, remove_last_point=True):
    # Load point cloud data
    point_cloud_array = load_point_cloud(file_path)
    point_cloud_data = array_to_dataframe(point_cloud_array)
    point_cloud_data = clean_data(point_cloud_data)

    # Interpolate profiles along wheel paths
    profiles = interpolate_profiles(point_cloud_data, spacing, num=720)

    # Filter out specific profiles
    filtered_profiles = filter_profiles(profiles, profiles_to_remove, remove_last_point=True)

    # Normalize profiles
    normalized_profiles = normalize_profiles(filtered_profiles)

    # Remove the last two points from each profile
    normalized_profiles = [profile[:-2] for profile in normalized_profiles]

    # Visualize the smoothed profiles
    visualize_profiles(normalized_profiles, "Input/Hwy 32/Track 1/outputs/Hwy32_Track1_Profiles")

    # Save the smoothed profiles to text files
    for i, profile in enumerate(normalized_profiles):
        filename = f"{output_dir}/Profile{i + 1}.txt"
        np.savetxt(filename, profile)


df = pd.read_csv('output_figures/Profiles/Track1_Profiles/IRI Results/output10.txt', sep='\s+')
# %% Hwy 32 Track 2 L
file_path = "Input/Hwy 32/Track 2 L/track2_leftturn_cleaned.ply"
output_dir = "Input/Hwy 32/Track 2 L/outputs"
profiles_to_remove = [0, 10]
spacing = 1  # to select the number of profiles. It is the spacing taken between the profiles along the road width


# filter_length = 0.25  # Moving average filter length in meters
# num_points_per_profile = 100  # Number of points in each profile
# profile_length = 130
# window_size = int(filter_length / (profile_length / num_points_per_profile))

def main(file_path, output_dir, profiles_to_remove, spacing, window_size, remove_last_point=True):
    # Load point cloud data
    point_cloud_array = load_point_cloud(file_path)
    point_cloud_data = array_to_dataframe(point_cloud_array)
    point_cloud_data = clean_data(point_cloud_data)

    # Interpolate profiles along wheel paths
    profiles = interpolate_profiles(point_cloud_data, spacing, num=500)

    # Filter out specific profiles
    filtered_profiles = filter_profiles(profiles, profiles_to_remove, remove_last_point=True)

    # Normalize profiles
    normalized_profiles = normalize_profiles(filtered_profiles)

    # Apply smoothing to the normalized profiles
    smoothed_profiles = []
    window_size_m = 0.25  # I applied a 250 mm moving average filter to profiles subsampled at 125 mm since it
    # resulted in smoother
    # profiles than directly subsampling at 250 mm.
    for profile in normalized_profiles:
        y_normalized = profile[:, 0]
        z_normalized = profile[:, 1]

        point_spacing = y_normalized[1] - y_normalized[0]
        window_size = int(window_size_m / point_spacing)

        if window_size >= 2:
            smoothed_z = np.convolve(z_normalized, np.ones(window_size) / window_size, mode='same')
            profile[:, 1] = smoothed_z

        smoothed_profiles.append(profile)

    # Remove the last two points from each profile
    smoothed_profiles = [profile[:-2] for profile in smoothed_profiles]

    # Visualize the smoothed profiles
    visualize_profiles(smoothed_profiles, "Input/Hwy 32/Track 2 L/outputs/Hwy32_Track2L_Profiles")
    # Save the smoothed profiles to text files
    for i, profile in enumerate(normalized_profiles):
        filename = f"{output_dir}/Profile{i + 1}.txt"
        np.savetxt(filename, profile)


df = pd.read_csv('output_figures/Profiles/Track2L/IRI Results/output9.txt', sep='\s+')
#%% Hwy 32 Track 2 R
file_path = "Input/Hwy 32/Track 2 R/track2_rightturn_cleaned.ply"
output_dir = "Input/Hwy 32/Track 2 R/outputs"
profiles_to_remove = [0,1,2,11,12]
spacing = 1  # to select the number of profiles. It is the spacing taken between the profiles along the road width


# filter_length = 0.25  # Moving average filter length in meters
# num_points_per_profile = 100  # Number of points in each profile
# profile_length = 130
# window_size = int(filter_length / (profile_length / num_points_per_profile))

def main(file_path, output_dir, profiles_to_remove, spacing, window_size, remove_last_point=True):
    # Load point cloud data
    point_cloud_array = load_point_cloud(file_path)
    point_cloud_data = array_to_dataframe(point_cloud_array)
    point_cloud_data = clean_data(point_cloud_data)

    # Interpolate profiles along wheel paths
    profiles = interpolate_profiles(point_cloud_data, spacing, num=760)

    # Filter out specific profiles
    filtered_profiles = filter_profiles(profiles, profiles_to_remove, remove_last_point=True)

    # Normalize profiles
    normalized_profiles = normalize_profiles(filtered_profiles)

    # Apply smoothing to the normalized profiles
    smoothed_profiles = []
    window_size_m = 0.75  # I applied a 250 mm moving average filter to profiles subsampled at 125 mm since it
    # resulted in smoother
    # profiles than directly subsampling at 250 mm.
    for profile in normalized_profiles:
        y_normalized = profile[:, 0]
        z_normalized = profile[:, 1]

        point_spacing = y_normalized[1] - y_normalized[0]
        window_size = int(window_size_m / point_spacing)

        if window_size >= 2:
            smoothed_z = np.convolve(z_normalized, np.ones(window_size) / window_size, mode='same')
            profile[:, 1] = smoothed_z

        smoothed_profiles.append(profile)

    # Remove the last two points from each profile
    smoothed_profiles = [profile[:-2] for profile in smoothed_profiles]
    smoothed_profiles = [profile[2:] for profile in smoothed_profiles]

    # Visualize the smoothed profiles
    visualize_profiles(smoothed_profiles, "Input/Hwy 32/Track 2 R/outputs/Hwy32_Track2R_Profiles")
    # Save the smoothed profiles to text files
    for i, profile in enumerate(normalized_profiles):
        filename = f"{output_dir}/Profile{i + 1}.txt"
        np.savetxt(filename, profile)


df = pd.read_csv('output_figures/Profiles/Track2R/IRI Results/output9.txt', sep='\s+')
