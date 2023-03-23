import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud(
    'Input/track2_pc.ply')


with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=2, min_points=50, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

#need to check the labels and filter the main las file according to the desired road cluster
unique, counts = np.unique(labels, return_counts=True)

# Combine unique values and their counts into a dictionary
counts_dict = dict(zip(unique, counts))
print(counts_dict)
#label 15 road. apply boolean filter

value = 5
bool_array = (labels == value)

track3_filtered = track3_elevation[bool_array]


track3_filtered.shape[0] - track3_elevation.shape[0]  # ok


_df_to_las_conversion(track3_filtered,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='track3_filtered', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])