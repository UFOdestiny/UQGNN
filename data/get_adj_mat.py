import geopandas
import numpy as np


# source: https://github.com/liyaguang/DCRNN/blob/master/scripts/gen_adj_mx.py
def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf

    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return adj_mx

#
# if __name__ == "__main__":
#     json_path = ""
#     save_path = ""
#
#     gdf = geopandas.GeoDataFrame.from_file(json_path)
#     ctr = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
#     N = len(ctr)
#
#     # bike_area = [4, 12, 13, 43, 45, 48, 50, 68, 79, 87, 88, 90, 100, 107, 113, 114, 125, 137, 140, 142, 143, 144, 148,
#     #              158, 161, 162, 163, 164, 170, 186, 209, 211, 224, 229, 230, 231, 232, 233, 234, 246, 249, 261]
#
#     sensor_ids_ = bike_area  # list(range(N))
#     distance = []
#     for i in bike_area:
#         for j in bike_area:
#             distance.append([i, j, ctr[i - 1].distance(ctr[j - 1])])
#
#     adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=sensor_ids_)
#     print(f"The shape of Adjacency Matrix: {adj_mx.shape}")
#     np.save(save_path, adj_mx)
