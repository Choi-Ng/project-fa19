import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tsp_solver.greedy import solve_tsp
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

# ======================================= optimization Helper ====================================

# merge two dropoffs by dropping off all the TAs that were originally dropped off in OLD_DROP location to NEw_DROP location
def mergeDropoffs(dropoff_to_TAs_dict, TA_to_dropoff_dict, old_drop, new_drop):
    list_TAs = dropoff_to_TAs_dict[old_drop]
    # 1. update the TA_to_dropoff_dict mapping
    for TA in list_TAs:
        TA_to_dropoff_dict[TA] = new_drop
    # 2a. add the TAs to the new_drop of dropoff_to_TAs_dict mapping
    if new_drop in dropoff_to_TAs_dict.keys():
        dropoff_to_TAs_dict[new_drop] = list_TAs
    #  2a. remove the TAs from the old_drop of dropoff_to_TAs_dict mapping
    else:
        dropoff_to_TAs_dict[new_drop].extend(list_TAs)
        dropoff_to_TAs_dict.pop(old_drop, None)
    return dropoff_to_TAs_dict, TA_to_dropoff_dict

# Find all dropoff locations along the tour except the starting_node of the tour
def dropoffs_along_the_tour(tour, dropoff_to_TAs_dict, TA_to_dropoff_dict):
    num_TAs_along_tour = 0
    dropoffs_along_tour = [stop for stop in tour if stop in dropoff_to_TAs_dict.keys() and stop != tour[0]]
    for dropoff in dropoffs_along_tour:
        num_TAs_along_tour += len(dropoff_to_TAs_dict[dropoff])
    return num_TAs_along_tour, dropoffs_along_tour

def start_and_end_indices_of_first_repeated_node(path, ignored_node):
    for node in path:
        if path.count(node) > 1 and node != ignored_node:
            reversedPath = path
            reversedPath.reverse()
            first_occurence = path.index(node)
            last_occurence = (len(reversedPath) - 1) - reversedPath.index(node)
            return True, first_occurence, last_occurence
    return False, None, None

# =========================== optimization algorithm ============================

def optimize_consecutive_tours(tour, dropoff_to_TAs_dict, TA_to_dropoff_dict):
    if len(tour) == 0:
        return tour, dropoff_to_TAs_dict, TA_to_dropoff_dict
    optimized_tour = []
    list_consecutive_tours = breakdown_consecutive_tours(tour)
    for subtour in list_consecutive_tours:
        optimized_subtour, dropoff_to_TAs_dict, TA_to_dropoff_dict = optimize_single_tour(subtour, dropoff_to_TAs_dict, TA_to_dropoff_dict)
        optimized_tour.extend(optimized_subtour)
    return optimized_tour, dropoff_to_TAs_dict, TA_to_dropoff_dict


def breakdown_consecutive_tours(tour):
    orig_tour = tour.copy()
    list_consecutive_tours = []
    starting_node = orig_tour[0]
    while orig_tour.count(starting_node) > 2:
        end_index = orig_tour.index(starting_node, 1)
        subtour = orig_tour[0: end_index] + [starting_node]
        orig_tour = orig_tour[end_index : len(orig_tour)]
        list_consecutive_tours.append(subtour)
    list_consecutive_tours.append(orig_tour)
    return list_consecutive_tours

def optimize_single_tour(tour, dropoff_to_TAs_dict, TA_to_dropoff_dict):

    starting_node = tour[0]
    num_TAs_along_tour, dropoffs_along_tour = dropoffs_along_the_tour(tour, dropoff_to_TAs_dict, TA_to_dropoff_dict)
    has_more_repeated_nodes, first_repeated_node_start, first_repeated_node_end = start_and_end_indices_of_first_repeated_node(tour, ignored_node=starting_node)

    # Base Case 1 - num_TAs_along_tour = 1
    if num_TAs_along_tour == 1:
        for dropoff in dropoffs_along_tour:
            dropoff_to_TAs_dict, TA_to_dropoff_dict = mergeDropoffs(dropoff_to_TAs_dict, TA_to_dropoff_dict, old_drop=dropoff, new_drop=starting_node)
        return [starting_node], dropoff_to_TAs_dict, TA_to_dropoff_dict

    # Base Case 2 - no any repeating nodes other than the starting_node
    if has_more_repeated_nodes == False:
        return tour, dropoff_to_TAs_dict, TA_to_dropoff_dict

    # Recursive Case
    optimized_tour = []
    remaining_path = tour
    while has_more_repeated_nodes:
        path_before_first_repeated_node = remaining_path[0 : first_repeated_node_start]
        first_consecutive_tours = remaining_path[first_repeated_node_start : first_repeated_node_end +1]
        path_after_first_repeated_node = remaining_path[first_repeated_node_end+1 : len(remaining_path)]
        # -----------------------------------------------------
        optimized_tour.extend(path_before_first_repeated_node)
        optimized_first_consecutive_tours, dropoff_to_TAs_dict, TA_to_dropoff_dict = optimize_consecutive_tours(first_consecutive_tours, dropoff_to_TAs_dict, TA_to_dropoff_dict)
        optimized_tour.extend(optimized_first_consecutive_tours)
        remaining_path = path_after_first_repeated_node
        # -----------------------------------------------------
        has_more_repeated_nodes, first_repeated_node_start, first_repeated_node_end = start_and_end_indices_of_first_repeated_node(tour, ignored_node=starting_node)
    if len(remaining_path) > 0:
        optimized_tour.extend(remaining_path)
    return optimized_tour, dropoff_to_TAs_dict, TA_to_dropoff_dict


# =================== Node Indexing Helpers ============================

def nameToIndex(homes, locations):
    return [locations.index(x) for x in homes]

def make_adjacency(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 'x':
                x[i][j] = 0
    return x

# ====================== TSP helpers =============================================

def makeHomesOnlyGraph(originalAdjacentMatrix, Hs):
    graph = csr_matrix(originalAdjacentMatrix)

    # Find shortest_path. The shortest dist_matrix is a complete graph since the original graph is connected
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, return_predecessors=True)
    homeOnly_adj_matrix, original_graph_index_key = removeNonHomes_rolsAndCols(dist_matrix, Hs)

    #shortest_paths =
    return homeOnly_adj_matrix, original_graph_index_key, predecessors

def removeNonHomes_rolsAndCols(dist_matrix, Hs):
    result_matrix = []
    original_graph_index_key = {}
    new_graph_index = {}
    for i in range(len(dist_matrix)):
        if i in Hs:
            row = []
            for j in range(len(dist_matrix[i])):
                if j in Hs:
                    row = row + [dist_matrix[i][j]]
            result_matrix = result_matrix + [row]
            original_graph_index_key[len(result_matrix)-1] = i
            new_graph_index[i] = len(result_matrix)-1

    return result_matrix, original_graph_index_key

def convertTSPnaivePath_to_originIndex(tsp_naive_path, original_graph_index_key):
    for i in range(len(tsp_naive_path)):
        tsp_naive_path[i] = original_graph_index_key[tsp_naive_path[i]]
    return tsp_naive_path

# ======================== Shortest Path Helpers =======================================

def replace_with_shortest_paths(orig_tour, sp_predecessors):

    # replace with shortest path
    replacedTour = []
    for i in range(len(orig_tour)-1):
        j = i + 1
        replacedTour = replacedTour + sp(sp_predecessors, orig_tour[i], orig_tour[j])
    return remove_consecutive_repeated_nodes(replacedTour)

def sp(predecessors, start, end):
    path = [predecessors[start][end], end]
    curr_vertex = predecessors[start][end]
    while(curr_vertex != -9999):
        curr_vertex = predecessors[start][curr_vertex]
        if curr_vertex != -9999:
            path.insert(0,curr_vertex)
    return path


def remove_consecutive_repeated_nodes(orig_tour):
    new_tour = [orig_tour[0]]
    for j in range(1, len(orig_tour)):
        i = j - 1
        if orig_tour[j] != orig_tour[i]:
            new_tour.append(orig_tour[j])
    return new_tour

#  ============ dropoff dicts Helpers =========================

def order_dict_in_order_of_tour(tour, orig_dict):
    result_dict = {}
    for stop in tour:
        if stop in orig_dict.keys():
            result_dict[stop] = orig_dict[stop]
    return result_dict

def naiveDropoffs(homes):
    dropoff_to_TAs_dict = {}
    TA_to_dropoff_dict = {}
    for h in homes:
        dropoff_to_TAs_dict[h] = [h]
        TA_to_dropoff_dict[h] = h
    return dropoff_to_TAs_dict, TA_to_dropoff_dict



def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    '''

    '''

    # Convert location names to node indices
    num_locations = len(list_of_locations)
    all_locations = [x for x in range(num_locations)]
    starting_node = list_of_locations.index(starting_car_location)
    homes = (nameToIndex(list_of_homes, list_of_locations))
    Hs = homes + [starting_node] if starting_node not in homes else homes
    Ls = [item for item in all_locations if item not in Hs]
    adjacency_matrix = make_adjacency(adjacency_matrix)

    # make a new graph G' with just the starting_node and the homes with edges of shortest_dist between them
    # also stored the shortest_path between those nodes in the original graph G in the predecessors mapping
    # Since the original graph G is connected, the new graph G' is a complete graph
    homeOnly_adj_matrix, original_graph_index_key, predecessors = makeHomesOnlyGraph(adjacency_matrix, Hs)

    # Find a tsp tour on the new graph G', using the original indexing in G on the tsp tour
    # and shift it to have the starting_node as the first stop
    tsp_naive = solve_tsp(homeOnly_adj_matrix)
    tsp_naive = convertTSPnaivePath_to_originIndex(tsp_naive, original_graph_index_key)
    tsp_naive = tsp_naive[tsp_naive.index(starting_node):len(tsp_naive)] +tsp_naive[0:tsp_naive.index(starting_node)] + [starting_node]

    # Replace any shortest_dist_edge that do not exist in the original grpah, with the shortest_path in G
    tour = replace_with_shortest_paths(tsp_naive, predecessors)

    # Create naive dropoff mappings: dropoff_to_TAs_dict, TA_to_dropoff_dict
    # where all TAs are dropped off at their homes
    dropoff_to_TAs_dict, TA_to_dropoff_dict = naiveDropoffs(homes)

    # optimize the tsp tour found by adding the dropoff option
    optimized_tour, dropoff_to_TAs_dict, TA_to_dropoff_dict = optimize_consecutive_tours(tour, dropoff_to_TAs_dict, TA_to_dropoff_dict)

    # Remove any consecutive repeated nodes in the tour created in optimization
    optimized_tour = remove_consecutive_repeated_nodes(optimized_tour)

    # Order the dropoff_to_TAs_dict mapping in the order of the tour
    dict = order_dict_in_order_of_tour(optimized_tour, dropoff_to_TAs_dict)

    return tour, dict

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
