# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:14:38 2025

@author: Seemab
"""

import itertools
import networkx as nx
import time

start_time = time.time()
def tree_adjacency_counts(T, n):
    """Return n x n integer matrix A where A[u][v] is multiplicity (loops on diag)."""
    A = [[0]*n for _ in range(n)]
    for u, v in T.edges():
        A[u][v] += 1
        A[v][u] += 1
    return A

def canonical_key_from_counts(A):
    """
    Canonical form by trying all relabelings (exact, but factorial in n).
    Works great for small n (like 5, 6, 7).
    """
    n = len(A)
    best = None
    for perm in itertools.permutations(range(n)):
        # build permuted upper-triangular including diagonal
        key = []
        for i in range(n):
            pi = perm[i]
            for j in range(i, n):
                pj = perm[j]
                key.append(A[pi][pj])
        key = tuple(key)
        if best is None or key < best:
            best = key
    return best

def count_unique_graphs(n, s, m):
    seen = set()

    # non-isomorphic unlabeled trees only (MUCH fewer than Prüfer)
    for T in nx.generators.nonisomorphic_trees(n):
        tree_edges = list(T.edges())  # (n-1) edges

        baseA = tree_adjacency_counts(T, n)

        # loops: multiset of vertices
        for loops in itertools.combinations_with_replacement(range(n), s):

            # extras: multiset of TREE edges (parallel edges on tree edges only)
            for extras in itertools.combinations_with_replacement(tree_edges, m):

                # copy base adjacency counts
                A = [row[:] for row in baseA]

                # add loops
                for v in loops:
                    A[v][v] += 1

                # add extra parallel edges on existing tree edges
                for u, v in extras:
                    A[u][v] += 1
                    A[v][u] += 1

                seen.add(canonical_key_from_counts(A))

    return len(seen)

if __name__ == "__main__":
    n, s, m = 4, 2, 3
    total = count_unique_graphs(n, s, m)
    print("Total unique non-isomorphic graphs =", total)
end_time = time.time()
print(f"Total computation time: {end_time - start_time:.4f} seconds\n")