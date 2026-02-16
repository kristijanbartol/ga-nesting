import numpy as np
import trimesh

def order_boundary_component_edges(edges: np.ndarray) -> np.ndarray:
    """
    Given an (m,2) int array of UNDIRECTED edges belonging to ONE connected boundary component,
    return ordered vertex indices along the boundary.
    Works for closed loops (every vertex degree==2) and open chains (two degree==1 endpoints).
    """
    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) == 0:
        return np.array([], dtype=np.int64)

    # adjacency: vertex -> list of neighbors
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    degrees = {v: len(nbrs) for v, nbrs in adj.items()}
    endpoints = [v for v, d in degrees.items() if d == 1]

    # start at an endpoint if open chain; otherwise any vertex (pick smallest for determinism)
    start = min(endpoints) if len(endpoints) >= 1 else min(adj.keys())

    ordered = [start]
    prev = -1
    cur = start

    # walk until we close (loop) or hit an endpoint (open chain)
    for _ in range(len(adj) + 5):  # safety bound
        nbrs = adj[cur]
        # choose next neighbor != prev
        nxt = None
        if len(nbrs) == 0:
            break
        if len(nbrs) == 1:
            nxt = nbrs[0]
        else:
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]

        if nxt == start:
            # closed loop
            break

        ordered.append(nxt)
        prev, cur = cur, nxt

        # if open chain and we reached the other endpoint, stop
        if len(endpoints) >= 2 and cur in endpoints and cur != start:
            break

    return np.asarray(ordered, dtype=np.int64)


def boundary_loops_from_edges(edges: np.ndarray) -> list[np.ndarray]:
    """
    Split boundary edges into connected components, and order each into a loop/chain.
    Returns a list of (k,) arrays of vertex indices.
    """
    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) == 0:
        return []

    comps = trimesh.graph.connected_components(edges)  # list of arrays of vertex ids
    loops = []
    for comp_verts in comps:
        comp_verts = np.asarray(comp_verts, dtype=np.int64)
        mask = np.isin(edges[:, 0], comp_verts) & np.isin(edges[:, 1], comp_verts)
        comp_edges = edges[mask]
        loops.append(order_boundary_component_edges(comp_edges))
    return loops


def polygon_area_2d(poly_xy: np.ndarray) -> float:
    """
    Shoelace area for an ordered polygon (not necessarily explicitly closed).
    """
    p = np.asarray(poly_xy, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
