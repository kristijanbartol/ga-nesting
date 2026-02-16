import sys

sys.path.append('/home/kristijan/LOOM/potpourri3d/src')

import numpy as np
import trimesh
from collections import defaultdict, Counter, deque
from itertools import combinations
import potpourri3d as pp3d

from geometry.const import EXCLUDE_PATCH_VIDXS


def perform_global_cut(paths_to_cut, vertices, faces):
    ref_vertices = vertices.copy()
    ref_faces = faces.copy()

    cut_vertices, cut_faces, cut_indices = _cut_paths(paths_to_cut, vertices, faces)

    compute_ref_embedding(ref_vertices, ref_faces, cut_vertices)
    
    adjacency_matrix = create_adjacency_matrix(faces)
    cut_mesh = trimesh.Trimesh(vertices=cut_vertices, faces=cut_faces, process=False)

    v_patch_idxs_dict, excluded_patch_idxs = flood_fill_vertex_patches_with_multilabels(cut_mesh, cut_indices)   
            
    patches, patch_faces, valid_patch_idxs, vertex_patch_index_map = extract_and_save_patch_meshes(cut_mesh, v_patch_idxs_dict, excluded_patch_idxs)
    
    seamlines_dict_list, symmetric_seamline_flags = extract_seamlines(patches, cut_indices, valid_patch_idxs, vertex_patch_index_map)
    
    # 1. Flatten index list
    all_indices = np.concatenate(cut_indices)

    # (optional) remove duplicates
    all_indices = np.unique(all_indices)

    # 2. Extract vertex coordinates
    points = cut_vertices[all_indices]   # shape: (M, 3)

    # 3. Create trimesh point cloud
    pc = trimesh.points.PointCloud(points)
    
    pc.export('cut_indices.ply')
    trimesh.Trimesh(vertices=cut_vertices, faces=cut_faces).export('cut_mesh.ply')

    return trimesh.Trimesh(vertices=cut_vertices, faces=cut_faces), patches, patch_faces, seamlines_dict_list, symmetric_seamline_flags, valid_patch_idxs


# cut all paths in the list of indices of paths_to_cut
def _cut_paths(kp_batch, vertices, faces):
    path_solver = pp3d.ExtendedEdgeFlipGeodesicSolver(vertices, faces)
    kp_coordinates = []

    for kpts in kp_batch:
        kp_coordinates.append([vertices[kpts[0]], vertices[kpts[-1]]])

    cut_vertices, cut_faces, cut_indices = path_solver.apply_cuts(kp_batch, kp_coordinates)

    return cut_vertices, cut_faces, cut_indices

def compute_ref_embedding(ref_vertices, ref_faces, query_points):
    ref_mesh = trimesh.Trimesh(ref_vertices, ref_faces, process=False)

    # closest point on reference surface for each query point
    closest_pts, dist, face_id = trimesh.proximity.closest_point(ref_mesh, query_points)

    # barycentric coords in that face
    tri = ref_mesh.triangles[face_id]                 # (N,3,3)
    bary = trimesh.triangles.points_to_barycentric(tri, closest_pts)  # (N,3)

    # optional: clamp small numerical negatives
    bary = np.clip(bary, -1e-8, 1.0 + 1e-8)

    #self.cut_ref_face_id = face_id.astype(np.int32)
    #self.cut_ref_bary = bary.astype(np.float32)


def create_adjacency_matrix(faces):
    adjacency_matrix = defaultdict(set)
    for face in faces:
        for i in range(3):
            vi = face[i]
            vj = face[(i + 1) % 3]
            adjacency_matrix[vi].add(vj)
            adjacency_matrix[vj].add(vi)
    return adjacency_matrix 


def assign_patch_labels(patches, garment_part, valid_patch_idxs, ref_point):
    symm_ref_point = ref_point.copy()
    symm_ref_point[0] *= -1    # reflect across X

    # the right point is the one with the smaller X coordinate and vice versa
    ref_point_right = ref_point if ref_point[0] < symm_ref_point[0] else symm_ref_point
    ref_point_left  = ref_point if ref_point[0] > symm_ref_point[0] else symm_ref_point

    patch_labels_dict = {
        'sleeve': [],
        'back': []
    }
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            # check whether the patch is part of the sleeve
            if garment_part == 'upper':
                count_right = (patch.vertices[:, 0] < ref_point_right[0]).sum()
                is_majority_right = count_right > (len(patch.vertices) / 2)
                count_left = (patch.vertices[:, 0] > ref_point_left[0]).sum()
                is_majority_left = count_left > (len(patch.vertices) / 2)

                if is_majority_right or is_majority_left:
                    patch_labels_dict['sleeve'].append(patch_idx)
            
            # check whether the patch is a back patch
            count_back = (patch.vertices[:, 2] < ref_point[2]).sum()
            is_majority_back = count_back > (len(patch.vertices) / 2)
            if is_majority_back:
                patch_labels_dict['back'].append(patch_idx)

    return patch_labels_dict


def flood_fill_vertex_patches_with_multilabels(mesh, polylines):
    '''
    Flood fill algorithm for (multi-)labeling vertices.

    Given a set of polylines ("list of lists of vertex indices"), floods the unvisited patches.
    The unvisited patch is found by iterating through all the vertices of the mesh, checking
    whether the vertex is already visited OR whether it's a boundary vertex, and if not,
    traverses the patch in a BFS fashion.
    '''
    boundary_set = set([x for xs in polylines for x in xs])
    V, F = mesh.vertices, mesh.faces

    # The adjacency dictionary is used for faster and more convenient traversal.
    adjacency = defaultdict(set)
    for face in F:
        for i in range(3):
            vi = face[i]
            vj = face[(i + 1) % 3]
            adjacency[vi].add(vj)
            adjacency[vj].add(vi)

    patch_idxs_dict = defaultdict(set)  # for each vertex, store a set of corresponding patch labels
    current_patch_idx = 0               # start with label=0 and increment when unexplored patch is found
    excluded_patch_idxs = set()         # the excluded patches are the ones that contain excluded vertices (predefined and fixed)

    # Some vertices remain unreached by traversal, yet surrounded by already-labeled vertices (boundaries).
    # To find such vertices, we check whether all the neighboring labels are the same.
    # Afterward, these vertices are labeled using the labels of their neighbors in a separate for loop below.
    def is_surrounded(v_start):
        neighbor_patch_idxs = [patch_idxs_dict[n] for n in adjacency[v_start]]
        return all(idx == neighbor_patch_idxs[0] and len(idx) == 1 for idx in neighbor_patch_idxs)

    for v_start in range(len(V)):
        # if on the boundary, or already labeled, or "surrounded", do not process (continue)
        if v_start in boundary_set or len(patch_idxs_dict[v_start]) > 0 or is_surrounded(v_start):
            continue
        queue = deque([v_start])
        patch_idxs_dict[v_start].add(current_patch_idx)
        touched_polylines = []      # record which polylines are "touched" so that we add the corresponding idxs later
        patch_vidxs = [v_start]     # separately record patch idxs to check whether it contains the excluded idxs

        while queue:
            v = queue.popleft()
            for nbr in adjacency[v]:
                # For the boundary vertices, do not label them now. Instead, record the whole polyline as "touched".
                if nbr in boundary_set:
                    touched_polylines_idxs = []
                    for polyline_idx, polyline in enumerate(polylines):
                        if nbr in polyline:
                            touched_polylines_idxs.append(polyline_idx)
                    # However, if the boundary vertex belongs to more than one polyline, do not label as "touched".
                    # Note that this is not a problem, since the polyline will be touched at some other location.
                    if len(touched_polylines_idxs) == 1:                        
                        touched_polylines.append(polylines[touched_polylines_idxs[0]])
                    continue

                # For the "normal" (non-boundary) neighbors, label them right away and add to the queue for traversal.
                if len(patch_idxs_dict[nbr]) == 0:
                    queue.append(nbr)
                    patch_idxs_dict[nbr].add(current_patch_idx)
                    patch_vidxs.append(nbr)
                    
        # For each touched polyline, label the corresponding vertices along the polylines (with the current label).
        # Note that, when done for multiple patches (labels), the boundary vertices will "naturally" have multiple labels.
        for touched_polyline in touched_polylines:
            for tv in touched_polyline:
                patch_idxs_dict[tv].add(current_patch_idx)

        # Finally, if the excluded vertex index is part of the patch, label the whole patch as excluded.
        for excluded_vidx in EXCLUDE_PATCH_VIDXS:
            if excluded_vidx in patch_vidxs:
                excluded_patch_idxs.add(current_patch_idx)

        current_patch_idx += 1

    # After the "regular" vertices are processed, the edge cases are the "surrounded" vertices that are now labeled.
    for v in range(len(V)):
        if len(patch_idxs_dict[v]) == 0:
            nbr = next(iter(adjacency[v]))
            patch_idxs_dict[v].add(nbr)

    return patch_idxs_dict, excluded_patch_idxs


def extract_patch(V, face_list):
    face_array = np.array(face_list)
    unique_verts, inverse_indices = np.unique(face_array.flatten(), return_inverse=True)

    V_patch = V[unique_verts]
    F_patch = inverse_indices.reshape((-1, 3))
    patch_mesh = trimesh.Trimesh(vertices=V_patch, faces=F_patch, process=False)

    return patch_mesh, unique_verts


def extract_and_save_patch_meshes(mesh, vertex_to_patch_idxs_dict, excluded_patch_idxs):
    '''
    Extract and save patches based on the flood fill vertex labels.

    In principle, the idea is to collect all the faces that belong to each patch label.
    Based on the selected faces, we find unique vertices and select the patch meshes.
    '''
    V, F = mesh.vertices, mesh.faces
    patch_faces = defaultdict(list)

    for face_idx, face in enumerate(F):
        v0, v1, v2 = face
        common_patch_idxs = set(vertex_to_patch_idxs_dict[v0]) & set(vertex_to_patch_idxs_dict[v1]) & set(vertex_to_patch_idxs_dict[v2])
        # When the face with multiple common labels is found, it certainly belongs to excluded patches (edge case).
        # In this case, we use an inner for loop and if statement to find any excluded label to use for this face.
        if len(common_patch_idxs) > 1:
            for excluded_patch_idx in excluded_patch_idxs:
                if excluded_patch_idx in common_patch_idxs:
                    patch_faces[excluded_patch_idx].append(face)
                    break
        else:
            for lbl in common_patch_idxs:
                patch_faces[lbl].append(face)

    patches = [trimesh.Trimesh()] * len(patch_faces)
    vertex_patch_index_map = dict()

    for patch_id, face_list in patch_faces.items():
        # Another edge case. This solution works for the current designs but is not general and could fail in the future.
        if len(face_list) < 20:
            excluded_patch_idxs.add(patch_id)

        patch_mesh, unique_verts = extract_patch(V, face_list)
        patches[patch_id] = patch_mesh

        # From the vertex indices from old to new, i.e., main mesh to patches.
        for local_idx, original_idx in enumerate(unique_verts):
            if original_idx not in vertex_patch_index_map:
                vertex_patch_index_map[original_idx] = {}
            vertex_patch_index_map[original_idx][patch_id] = local_idx

    # Finally, store valid patch labels for later processing.
    valid_patch_idxs = set(range(len(patch_faces))) - set(excluded_patch_idxs)
    
    return patches, patch_faces, valid_patch_idxs, vertex_patch_index_map


def extract_seamlines(patches, boundary_indices_array, valid_patch_idxs, vertex_patch_index_map):
    '''
    Extract seamline indices as pairs of corresponding vertices in the neighboring patches.

    Each seamline is a separate entry and always belongs to a single pair of patches (although not vice versa).
    The boundaries (other) are on the border of the garment and connect with the excluded patches (e.g., face etc.).
    
    vertex_patch_index_map: {vidx: {label: patch_vidx}}, e.g., {115: {3: 312, 4: 1117, 6: 2}}
    seamlines_dict_list: [{(patch_idx1, patch_idx2): [(vidx_patch1, vidx_patch2)]}]
    '''
    seamlines_dict_list = []
    for boundary_indices in boundary_indices_array:
        seamlines_dict = defaultdict(list)
        is_seamline = True

        for vidx in boundary_indices:
            v_patch_idxs = set(vertex_patch_index_map[vidx].keys())
            #filtered_patch_idxs = sorted(set(v_patch_idxs) & valid_patch_idxs)
            filtered_patch_idxs = sorted(set(v_patch_idxs) & valid_patch_idxs)
            #if len(filtered_patch_idxs) == 1:    # then it's a boundary, not a seamline
            if len(filtered_patch_idxs) == 0:    # then it's a boundary, not a seamline
                is_seamline = False
                break
            patch_pairs = list(combinations(filtered_patch_idxs, 2))

            for patch_pair in patch_pairs:
                patch1_idx = vertex_patch_index_map[vidx][patch_pair[0]]
                patch2_idx = vertex_patch_index_map[vidx][patch_pair[1]]
                seamlines_dict[patch_pair].append((patch1_idx, patch2_idx))

        # After collecting all the seamlines, there are tips of seamlines, connected via either one or two vertices.
        # Although these are logically valid connections, they are not useful for the energy minimization.
        for patch_pair in list(seamlines_dict.keys()):
            if len(seamlines_dict[patch_pair]) <= 2:
                del(seamlines_dict[patch_pair])

        # Finally, add only the seamlines (not other boundaries).
        if is_seamline and len(seamlines_dict) > 0:
            seamlines_dict_list.append(seamlines_dict)
        
    symmetric_seamline_flags = [False] * len(seamlines_dict_list)
    for seam_idx in range(len(seamlines_dict_list)):
        patch_pair, vertex_pairs_list = next(iter(seamlines_dict_list[seam_idx].items()))    # take the one and only seam (using dictionary to have the label pair (patch1_idx, patch2_idx))
        patch_element = 0   # use the first patch for fetching the vertex indices and using the patch idx
        patch_idx = patch_pair[patch_element]
        verts = np.array([patches[patch_idx].vertices[vertex_pair[patch_element]] for vertex_pair in vertex_pairs_list])
        if np.mean(np.abs(verts[:, 0]) < 9e-3) > 0.9:
            symmetric_seamline_flags[seam_idx] = True

    return seamlines_dict_list, symmetric_seamline_flags
