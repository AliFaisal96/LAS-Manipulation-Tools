import pandas as pd
import numpy as np
import laspy
from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d

from src.manipulation_tools import _df_to_las_conversion

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def _las_to_df(las_file: laspy.LasData) -> pd.DataFrame:
    data = []
    columns = list(las_file.point_format.dimension_names)
    columns[:3] = ['x','y','z']
    for column in columns:
        data.append(np.array(las_file[column]))
    df = pd.DataFrame(np.array(data).T, columns=columns)
    return df


# compressing a LAS file into LAZ and vise versa
def _las_to_laz_conversion(las_file: laspy.LasData, address: str, name: str, do_compress: bool = True):
    if do_compress:
        las_file.write(do_compress=do_compress, destination=Path(f'{address}') / f'{name}.laz')
    else:
        las_file.write(do_compress=do_compress, destination=Path(f'{address}') / f'{name}.las')



if __name__ == '__main__':
    las_file = laspy.read(
        'C:/Users/unbre/OneDrive - UBC/Ph.D. Work/PCTool/PointCloudTool-master/Ledcor-Highway32/Track_H_20221114_192938 Profiler.zfs_0_annotated.laz')
    df = _las_to_df(las_file)
    # df = pd.read_csv('../data/testdf.csv')
    print(df)
    # _write_df_to_csv(df, '../data/', 'ttlaz')
    # _laz_to_las_conversion(las_file, '../data/', 'ttlas')
    # _df_to_las_conversion(df, address='../data/', name='ttlas3',
    #                       data_columns=['X', 'Y', 'Z', 'intensity', 'classification'])

# lets just export the df as is with colors
# TESTING
# _df_to_las_conversion(df,
#                       address='C:/Users/unbre/OneDrive - UBC/Ph.D. Work/PointCloudTool/PointCloudTool-master/PointCloudTool-master',
#                       name='df', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
#         , 'red', 'green', 'blue'])

np.max(df['Z']), np.min(df['Z'])
(df['Z']).describe()

# df_intensityfilter = df[(df['intensity'] > 1) & (df['intensity'] <= 30)]

# _df_to_las_conversion(df_intensityfilter,
#                       address='C:/Users/unbre/OneDrive - UBC/Ph.D. Work/PointCloudTool/PointCloudTool-master/PointCloudTool-master',
#                       name='df_intensity', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
#         , 'red', 'green', 'blue'])


df_elevation = df[df['Z'] < 8000]
df_elevation2= df[(df['Z'] > 6900) & (df['Z'] <= 7200)]

_df_to_las_conversion(df_elevation2,
                      address='C:/Users/unbre/OneDrive - UBC\Ph.D. Work\PCTool\PointCloudTool-master',
                      name='df_elevation2', data_columns=['X', 'Y', 'Z', 'intensity', 'classification'
        , 'red', 'green', 'blue'])

# On CloudCompare, I tried using statistical outlier filter to remove points that are scattered with no
# spatial proximity to other points but parts of the road were being removed so lets try
# the open3d library using DBSCAN

pcd = o3d.io.read_point_cloud(
    'C:/Users/unbre/OneDrive - UBC/Ph.D. Work/PointCloudTool/PointCloudTool-master/PointCloudTool-master/df.ply')

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=2, min_points=20, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd])

