# project-fa19
CS 170 Fall 2019 Project

1) TSP library
Make sure the greedy.py file is in a folder called tsp_solver. That is the tsp library we used in our code.
If the tsp_solver/greedy.py file could not be found in our code submission. Please download it from https://github.com/dmishin/tsp-solver.
Just download the greedy.py file and put it under a folder called tsp_solver.
This import statement in the beginning of solver.py has imported this TSP Library we used
>>> from tsp_solver.greedy import solve_tsp

2) Shortest Path Library
https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
The two import statements in the beginning of solver.py has imported this Shortest Path Library we used
>>> from scipy.sparse import csr_matrix
>>> from scipy.sparse.csgraph import shortest_path

python3 solver.py --all inputs/ outputs/
