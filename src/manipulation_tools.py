import laspy
import pandas as pd
import numpy as np
from pathlib import Path

from laspy import PointFormat
from laspy.header import Version

from src.LAS_tools import _las_to_df

_standard_classes = {'never classified': [0], 'unclassified': [1], 'ground': [2], 'low vegetation': [3],
                     'medium vegetation': [4], 'high vegetation': [5], 'building': [6], 'low point': [7],
                     'high point': [8], 'water': [9], 'rail': [10], 'road surface': [11], 'bridge deck': [12],
                     'wire-guard': [13], 'wire-conductor': [14], 'transmission tower': [15],
                     'wire-structure connector': [16]}


# converting the extracted point cloud data in the form of pandas dataframe into a CSV file
def _write_df_to_csv(df: pd.DataFrame, address: str, name: str):
    df.to_csv(Path(f'{address}') / f'{name}.csv')


def _json_class_mapping(file):
    pass


def _las_class_standardization(df: pd.DataFrame, classes_file: str = None, classes_json: object = None,
                               classes_dict: dict = None) -> pd.DataFrame:
    if classes_dict is not None:
        mapping_dict = {}
        for key in classes_dict:
            mapping_dict[classes_dict[key]] = _standard_classes[key][0]
        # this line of code maps the initial classes to the standard ones. If there is a missing class it will fill
        # it with 1 which is the value for unclassified data
        df['classification'] = df['classification'].map(mapping_dict).fillna(1)
    return df


# reducing the intensity of point cloud by dropping points at random based on a percentage of points or their count
def _df_intensity_reduction(df: pd.DataFrame, percentage: int = None, count: int = None) -> pd.DataFrame:
    if percentage is not None:
        remove_n = len(df) // 100 * percentage
    elif count is not None:
        remove_n = count
    else:
        raise Exception("Either count of the points or the percentage of dropped points should be set")
    drop_indices = np.random.choice(df.index, remove_n, replace=False)
    df_subset = df.drop(drop_indices)
    return df_subset


def _df_to_las_conversion(df: pd.DataFrame, address: str, name: str, las_version: tuple = (1, 4), las_format: int = 7,
                          data_columns:
                          list[str] = [], scales=[0.0001, 0.0001, 0.0001]):
    header = laspy.header.LasHeader(version=Version(las_version[0], las_version[1]),
                                    point_format=PointFormat(las_format))
    XYZ = np.floor(df[['x', 'y', 'z']].values / scales)
    mins = np.min(np.floor(df[['x', 'y', 'z']].values), axis=0) // 1000 * 1000
    header.offset = mins
    header.scale = scales
    las = laspy.LasData(header)
    XYZ -= mins
    # las['X'] = XYZ[:, 0]
    # las['Y'] = XYZ[:, 1]
    # las['Z'] = XYZ[:, 2]
    for column in data_columns:
        las[column] = df[column].values
    las.write(str(Path(f'{address}') / f'{name}.las'))
    print(f'The LAS file has been created at address {Path(address)}')


if __name__ == '__main__':
    # test_classes = {'rail': 3, 'building': 1, 'water': 2}
    # df = _las_class_standardization(pd.read_csv('../data/testdf.csv'), classes_dict=test_classes)
    # print(df.classification)
    las_file = laspy.read('../../datasets/Tile 2 6.laz')
    df = _las_to_df(las_file)
    _df_to_las_conversion(df, address='../../datasets', name='Test',
                          data_columns=['x','y','z','intensity', 'return_number', 'number_of_returns',
                                        'synthetic', 'key_point', 'withheld', 'overlap', 'scanner_channel',
                                        'scan_direction_flag', 'edge_of_flight_line', 'classification',
                                        'user_data', 'scan_angle', 'point_source_id', 'gps_time', 'red',
                                        'green', 'blue'])
