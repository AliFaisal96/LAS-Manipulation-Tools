import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def array_to_dataframe(point_cloud_array):
    data = pd.DataFrame(point_cloud_array, columns=["X", "Y", "Z"])
    return data


def interpolate_profiles(data, spacing, wheel_paths=True):
    if wheel_paths:
        y_min, y_max = data["Y"].min(), data["Y"].max()
        profiles = []

        for x in np.arange(data["X"].min(), data["X"].max(), spacing):
            profile_data = data[(data["X"] >= x) & (data["X"] < x + spacing)].sort_values(by="Y")
            unique_y_count = len(profile_data["Y"].unique())

            if not profile_data.empty and unique_y_count >= 2:
                y = profile_data["Y"].values
                z = profile_data["Z"].values
                y_new = np.linspace(y_min, y_max, num=100)
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
        filename = f"Track 1_Profile{i+1}.txt"
        np.savetxt(filename, profile)
def visualize_profiles(profiles):
    plt.figure(figsize=(10, 6))

    for i, profile in enumerate(profiles):
        plt.plot(profile[:, 0], profile[:, 1], label=f'Profile {i + 1}')

    plt.xlabel('Longitudinal Length (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Longitudinal Profiles')
    plt.legend()
    plt.show()

os.getcwd()
file_path = "Input/Alberta/Road_L1L2.ply"
output_dir = "Outputs/"
profiles_to_remove = [0,6]
spacing = 1.5
window_size = 1


def main(file_path, output_dir, profiles_to_remove, spacing, window_size, remove_last_point=True):
    # Load point cloud data
    point_cloud_array = load_point_cloud(file_path)
    point_cloud_data = array_to_dataframe(point_cloud_array)
    point_cloud_data = clean_data(point_cloud_data)

    # Interpolate profiles along wheel paths
    profiles = interpolate_profiles(point_cloud_data, spacing)

    # Filter out specific profiles
    filtered_profiles = filter_profiles(profiles, profiles_to_remove, remove_last_point=True)

    # Normalize profiles
    normalized_profiles = normalize_profiles(filtered_profiles)

    # Apply moving average filter to the normalized profiles
    smoothed_profiles = []
    for profile in normalized_profiles:
        smoothed_profile = np.zeros_like(profile)
        smoothed_profile[:, 0] = profile[:, 0]
        smoothed_profile[:, 1] = np.convolve(profile[:, 1], np.ones(window_size) / window_size, mode='same')
        smoothed_profiles.append(smoothed_profile)

    # Remove the last two points from each profile
    smoothed_profiles = [profile[:-2] for profile in smoothed_profiles]

    # Visualize the smoothed profiles
    visualize_profiles(smoothed_profiles)

    # Save the smoothed profiles to text files
    for i, profile in enumerate(smoothed_profiles):
        filename = f"{output_dir}/Profile{i+1}.txt"
        np.savetxt(filename, profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point cloud data and generate longitudinal profiles.")
    parser.add_argument("--input-file", required=True, help="Path to the input point cloud file")
    parser.add_argument("--output-dir", required=True, help="Path to the output directory")
    parser.add_argument("--remove", nargs="+", type=int, default=[0, 11], help="Indices of profiles to remove")
    parser.add_argument("--spacing", type=int, default=1, help="Spacing between profile sections")
    parser.add_argument("--window_size", type=int, default=3, help="Window size for the moving average filter")
    parser.add_argument("--remove_last_point", action="store_true", default=False, help="Remove the last point in each profile")
    args = parser.parse_args()

    main(args.input_file, args.output_dir, args.remove, args.spacing, args.window_size, args.remove_last_point)




df = pd.read_csv('Outputs/Profiles/BCMoTI_Profiles/Section 2/Results/output3.txt', sep='\s+')
iri_values = df['IRI value']
