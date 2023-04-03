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