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
def nameToIndex(homes, locations):
    return [locations.index(x) for x in homes]

def make_adjacency(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 'x':
                x[i][j] = 0
    return x


def makeHomesOnlyGraph(originalAdjacentMatrix, Hs):
    graph = csr_matrix(originalAdjacentMatrix)

    # Find shortest_path. The shortest dist_matrix is a complete graph since the original graph is connected
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, return_predecessors=True)
    homeOnly_adj_matrix, original_graph_index_key = removeNonHomes_rolsAndCols(dist_matrix, Hs)

    #shortest_paths =
    return homeOnly_adj_matrix, original_graph_index_key, predecessors

def removeNonHomes_rolsAndCols(dist_matrix, Hs):
    result_matrix = []
    original_graph_index = {}
    new_graph_index = {}
    for i in range(len(dist_matrix)):
        if i in Hs:
            row = []
            for j in range(len(dist_matrix[i])):
                if j in Hs:
                    row = row + [dist_matrix[i][j]]
            result_matrix = result_matrix + [row]
            original_graph_index[len(result_matrix)-1] = i
            new_graph_index[i] = len(result_matrix)-1

    return result_matrix, original_graph_index_key

def convertTSPnaivePath_to_originIndex(tsp_naive_path, original_graph_index_key):
    for i in range(len(tsp_naive_path)):
        tsp_naive_path[i] = original_graph_index_key[tsp_naive_path[i]]
    return tsp_naive_path


def sp(predecessors, start, end):
    path = [predecessors[start][end], end]
    curr_vertex = predecessors[start][end]
    while(curr_vertex != -9999):
        curr_vertex = predecessors[start][curr_vertex]
        path.insert(0,curr_vertex)
    return path

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

    homeOnly_adj_matrix, original_graph_index_key, predecessors = makeHomesOnlyGraph(adjacency_matrix, Hs)



    tsp_naive = solve_tsp(homeOnly_adj_matrix)
    tsp_naive = convertTSPnaivePath_to_originIndex(tsp_naive_path, original_graph_index_key)
    tsp_naive = tsp_naive[tsp_naive.index(starting_index):len(tsp_naive)] +tsp_naive[0:tsp_naive.index(starting_index)] + [0]
    #print(tsp_naive)
    for i in range(len(tsp_naive)-1):
        j = i + 1
        if(adjacency_matrix[tsp_naive[i]][tsp_naive[i+1]]==0):
            #print(sp(predecessors,i,j))
            storesp = sp(predecessors,i,j)[2:-1]
            if (storesp != None):
                for k in range(len(storesp)):
                    tsp_naive.insert(i+1+k, storesp[k])
            #tsp_naive.insert(i+1,sp(predecessors,i,j)[2:-1])
    #tsp_naive = lambda l: [item for sublist in row for item in sublist]
    #print(tsp_naive)
    dict ={}
    for h in tsp_naive:
        if(h in homes):
            dict[h] = [h]
    print(dict)
    return tsp_naive, dict

    exit()
    pass

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
