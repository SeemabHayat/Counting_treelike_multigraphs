# Counting tree-like multigraphs

A Python implementation using Dynamic Programming to count non-isomorphic tree-like multigraphs with `n` vertices, `s` self-loops, and `m` multiple edges.

Requirements
- Python 3
- NumPy (`pip install numpy`)

Usage
1. Open the script and edit the parameter limits at the top:
   ```python
   nv = 4  # max n (vertices)
   ns = 2  # max s (self-loops)
   nm = 3  # max m (multiple edges)
# For NetworkX
A Python script to  count non-isomorphic tree-like multigraphs with `n` vertices, `s` self-loops, and `m` multiple edges. Intended as a **sanity-check** for the DP algorithm on small inputs.

 Requirements
- Python 3
- NetworkX (`pip install networkx`)

Usage
1. Edit the parameters at the bottom of the script:
   ```python
   n, s, m = 4, 2, 3
