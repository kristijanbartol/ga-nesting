# Based on the newton/examples/cloth/example_cloth_style3d.py

# Notes for sanity check:
#   - the sewing pattern meshes should be facing the camera (back patches should be reflected to front)
#   - the body mesh should match the 3D patches
#   - the sewing pattern should have all the triangles facing the same direction (i.e., horse patterns were a bit problematic)
# Otherwise, the CUDA error might occur due to "inverted or degenerate triangles".

import os
import numpy as np
import warp as wp
import trimesh
from pathlib import Path

import newton
import newton.examples
from newton import Mesh, ParticleFlags

from geometry.param_mesh_uv import add_uv_coordinates, trimesh_to_plydata


PANTS = False
HORSE = False


def read_patches(is_adapt=False, garment_type='upper', patches_dir='data/patches/upper'):
    patch_meshes = []
    for patch_dirname in sorted(os.listdir(patches_dir)):
        patch_dir = f'{patches_dir}/{patch_dirname}/'
        if not is_adapt:
            patch_fname = [x for x in os.listdir(patch_dir) if x[-3:] == 'ply'][0]
        else:
            patch_fname = [x for x in os.listdir(patch_dir) if 'target' in x][0]
        patch_mesh = trimesh.load(os.path.join(patch_dir, patch_fname))
        patch_mesh = patch_mesh.subdivide()
        patch_meshes.append(patch_mesh)
    merged_mesh = trimesh.util.concatenate(patch_meshes)
    merged = trimesh.Trimesh(vertices=merged_mesh.vertices, faces=merged_mesh.faces, process=True)
    unmerged = trimesh.Trimesh(vertices=merged_mesh.vertices, faces=merged_mesh.faces, process=False)
    return merged, unmerged


def read_back_idxs(garment_type='upper'):
    with open(f'data/labels/{garment_type}/back.txt', 'r') as labels_f:
        back_idxs = list(map(int, labels_f.read().split()))
    return back_idxs


def _outer_boundary_centroid_mm(mesh):
    """Centroid of the outer boundary loop in mm (same as Stage2 preprocessing)."""
    from nesting.utils import boundary_loops_from_edges, polygon_area_2d
    V2 = mesh.vertices[:, :2] * 1000.0
    ue = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    be = mesh.edges[ue]
    loops = boundary_loops_from_edges(be)
    areas = [abs(polygon_area_2d(V2[lp])) for lp in loops]
    outer = loops[int(np.argmax(areas))]
    return V2[outer].mean(axis=0)


def read_sewing_pattern(pattern_root: str = 'results/pattern/latest', garment_type='upper'):
    """Returns the merged 2D simulation mesh and patch_info list.

    patch_info: list of (patch_id, vertex_start, vertex_end, was_y_flipped, boundary_centroid_mm)
      - vertex ranges index into the merged UV array
      - boundary_centroid_mm matches the Stage2 solver's centering convention
    """
    param_2d_dir = os.path.join(pattern_root, garment_type)
    back_idxs = read_back_idxs(garment_type)
    patch_2d_meshes = []
    patch_info = []
    vertex_offset = 0
    for patch_dirname in sorted(os.listdir(param_2d_dir)):
        patch_dir = f'{param_2d_dir}/{patch_dirname}/'
        patch_idx = int(patch_dirname[-2:])
        param_2d_mesh = trimesh.load(os.path.join(patch_dir, 'optim_final-seams.ply'))
        # Boundary centroid from un-flipped mesh, same as Stage2 preprocessing
        bm_mm = _outer_boundary_centroid_mm(param_2d_mesh)
        flipped = patch_idx in back_idxs
        if flipped:
            param_2d_mesh.vertices[:, 1] *= -1
        param_2d_mesh = param_2d_mesh.subdivide()
        n = len(param_2d_mesh.vertices)
        patch_info.append((patch_idx, vertex_offset, vertex_offset + n, flipped, bm_mm))
        vertex_offset += n
        patch_2d_meshes.append(param_2d_mesh)
    merged_mesh = trimesh.util.concatenate(patch_2d_meshes)
    return trimesh.Trimesh(vertices=merged_mesh.vertices, faces=merged_mesh.faces, process=False), patch_info


class HeadlessViewer:
    def set_model(self, model):
        self.model = model

    def apply_forces(self, state):
        # no user interaction / wind / mouse forces
        pass


def extract_upper_rim(garment_verts, garment_faces):
    """
    Extracts the upper rim vertices (e.g., waistline) from a triangle mesh.
    
    Parameters:
        garment_verts (numpy.ndarray): Array of vertex positions (N x 3).
        garment_faces (numpy.ndarray): Array of triangle faces (M x 3), with indices into `vertices`.
    
    Returns:
        numpy.ndarray: Indices of the vertices forming the upper rim.
    """
    # Calculate the center of mass (COM) of the mesh
    center_of_mass = np.mean(garment_verts, axis=0)
    
    # Filter vertices above the center of mass
    above_com_indices = np.where(garment_verts[:, 1] > center_of_mass[1])[0]
    
    # Extract edges from faces
    edges = np.vstack([
        garment_faces[:, [0, 1]],
        garment_faces[:, [1, 2]],
        garment_faces[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)  # Ensure consistent edge ordering
    
    # Find unique edges and their counts
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    
    # Boundary edges appear only once
    boundary_edges = unique_edges[counts == 1]
    
    # Filter boundary edges to only include vertices above the center of mass
    rim_edges = boundary_edges[
        np.all(np.isin(boundary_edges, above_com_indices), axis=1)
    ]
    
    # Flatten the rim edges to get unique rim vertices
    rim_vertices = np.unique(rim_edges)
    
    return rim_vertices.tolist()


class Example:

    def __init__(self, viewer, avatar, garment_type, is_adapt=False,
                 tsol=None, kappas_by_id=None, K=None,
                 period_u_mm=None, period_v_mm=None,
                 pattern_root='results/pattern/latest',
                 patches_dir='data/patches/upper'):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        # must be an even number when using CUDA Graph
        self.sim_substeps = 4
        self.sim_time = 0.0
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.iterations = 10

        self.viewer = viewer
        builder = newton.Style3DModelBuilder(up_axis=newton.Axis.Z)

        #self.scale = 10     # scale up to improve simulation result
        self.scale = 5     # scale up to improve simulation result

        garment_mesh_3d, garment_mesh_3d_unmerged = read_patches(is_adapt, garment_type, patches_dir=patches_dir)
        garment_mesh_2d, patch_info = read_sewing_pattern(pattern_root, garment_type)

        if PANTS:
            rim_idxs = extract_upper_rim(garment_mesh_3d.vertices, garment_mesh_3d.faces)

        assert len(garment_mesh_3d.faces) == len(garment_mesh_2d.faces)

        self.cloth_faces = garment_mesh_3d.faces.copy()   # (F, 3) int

        # UV export: store unmerged faces and UV coords
        assert len(garment_mesh_3d_unmerged.vertices) == len(garment_mesh_2d.vertices), (
            f"UV vertex mismatch: {len(garment_mesh_3d_unmerged.vertices)} vs {len(garment_mesh_2d.vertices)}"
        )
        self.unmerged_faces = garment_mesh_3d_unmerged.faces.copy()
        self.garment_mesh_uv = self._build_fabric_uv(
            garment_mesh_2d.vertices[:, :2], patch_info,
            tsol, kappas_by_id, K, period_u_mm, period_v_mm
        )

        # Map each unmerged vertex -> its merged vertex index (for position propagation)
        groups = trimesh.grouping.group_rows(garment_mesh_3d_unmerged.vertices, digits=6)
        unmerged_to_merged_dict = {}
        for group in groups:
            merged_idx = int(garment_mesh_3d.kdtree.query(garment_mesh_3d_unmerged.vertices[group[0]])[1])
            for idx in group:
                unmerged_to_merged_dict[idx] = merged_idx
        N_unmerged = len(garment_mesh_3d_unmerged.vertices)
        self.unmerged_to_merged = np.array([unmerged_to_merged_dict[i] for i in range(N_unmerged)], dtype=np.int32)
        self.save_every = 10
        self.frame_idx = 0
        self.out_dir = Path(f"./results/simulation/{garment_type}/")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        #garment_mesh_3d.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 3, [0, 1, 0]))

        garment_mesh_points = garment_mesh_3d.vertices * self.scale
        #garment_mesh_points[:, 1] += self.scale
        garment_mesh_indices = garment_mesh_3d.faces.flatten()

        garment_mesh_uv = garment_mesh_2d.vertices[:, :2] * self.scale
        garment_mesh_uv_indices = garment_mesh_2d.faces.flatten()

        if type(avatar) == str:
            avatar_mesh = trimesh.load(avatar)
        else:
            avatar_mesh = avatar
        #avatar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 3, [0, 1, 0]))
        avatar_mesh_points = np.asarray(avatar_mesh.vertices, dtype=np.float32) * self.scale
        #avatar_mesh_points[:, 1] += self.scale
        avatar_mesh_indices = np.asarray(avatar_mesh.faces, dtype=np.int32)

        if HORSE:
            '''
            avatar_mesh_points[:, 1] += 0.515 * self.scale
            #avatar_mesh_points[:, 2] += 1.353 * self.scale
            avatar_mesh_points[:, 2] += 1.355 * self.scale
            '''
            
            avatar_mesh_points[:, 0] += 0.29 * self.scale
            avatar_mesh_points[:, 1] += 0.385 * self.scale
            avatar_mesh_points[:, 2] += 0.9 * self.scale
            
        builder.add_aniso_cloth_mesh(
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            vel=wp.vec3(0.0, 0.0, 0.0),
            tri_aniso_ke=wp.vec3(1.0e3, 1.0e3, 1.0e2),
            edge_aniso_ke=wp.vec3(2.0e-7, 1.0e-7, 5.0e-8),
            panel_verts=garment_mesh_uv.tolist(),
            panel_indices=garment_mesh_uv_indices.tolist(),
            vertices=garment_mesh_points.tolist(),
            indices=garment_mesh_indices.tolist(),
            density=1.0,
            scale=1.0,
            particle_radius=3.5e-2 * self.scale / 5
        )

        builder.add_shape_mesh(
            body=builder.add_body(),
            xform=wp.transform(
                p=wp.vec3(0, 0, 0),
                q=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
            ),
            mesh=Mesh(avatar_mesh_points, avatar_mesh_indices),
        )

        
        if is_adapt:
            pants_mesh = trimesh.load('designs/teaser-pants00/simulation/adapt/sim.ply')
            pants_points = np.asarray(pants_mesh.vertices, dtype=np.float32) * self.scale
            #avatar_mesh_points[:, 1] += self.scale
            pants_indices = np.asarray(pants_mesh.faces, dtype=np.int32)
            builder.add_shape_mesh(
                body=builder.add_body(),
                xform=wp.transform(
                    p=wp.vec3(0, 0, 0),
                    q=wp.quat_from_axis_angle(axis=wp.vec3(1, 0, 0), angle=wp.pi / 2.0),
                ),
                mesh=Mesh(pants_points, pants_indices),
            )
        
        
        fixed_points = []
        if PANTS:
            fixed_points = rim_idxs

        # add a table
        #builder.add_ground_plane()
        self.model = builder.finalize()

        # set fixed points
        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in fixed_points:
            flags[fixed_vertex_id] = flags[fixed_vertex_id] & ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        #pr = 3.5e-2 * self.scale / 5   # same value you pass to add_aniso_cloth_mesh
        #self.model.soft_contact_radius = pr
        #self.model.soft_contact_margin = 0.5 * pr
        # Also: make contact stiffer (penetration goes down as stiffness/iterations go up)
        #self.model.soft_contact_ke = 1.0e3 * self.scale   # try 1e3..1e5 *scale
        #self.model.soft_contact_kd = 1.0e-5 * self.scale  # damping (tune)

        # set up contact query and contact detection distances
        self.model.soft_contact_radius = 0.2e-2 * self.scale
        self.model.soft_contact_margin = 0.35e-2 * self.scale
        self.model.soft_contact_ke = 1.0e1 * self.scale
        self.model.soft_contact_kd = 1.0e-6 * self.scale
        self.model.soft_contact_mu = 0.2e-3 * self.scale
        if not HORSE:
            self.model.set_gravity((0.0, 0.0, -9.81))
        else:
            self.model.set_gravity((0.0, 9.81, 0))
        #self.model.set_gravity((0.0, 0.0, -0.))

        self.solver = newton.solvers.SolverStyle3D(
            model=self.model,
            iterations=self.iterations,
        )
        self.solver.precompute(
            builder,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    @staticmethod
    def _build_fabric_uv(raw_uv, patch_info, tsol, kappas_by_id, K, period_u_mm, period_v_mm):
        """Apply Stage2 transforms + kappa phase offsets to get fabric-space UV.

        raw_uv: (N, 2) array in metres from the 2D parameterisation (may have
                y-flipped rows for back patches — un-flipped here before transform).
        patch_info: list of (patch_id, start, end, was_y_flipped, boundary_centroid_mm).
        Returns UV in mm in fabric space, ready for Polyscope texture mapping after
        dividing by (period_u_mm, period_v_mm).
        """
        from nesting.phase_utils import Rigid2D

        if tsol is None:
            return raw_uv * 1000.0

        fabric_uv = np.empty_like(raw_uv)
        for pid, start, end, was_flipped, bm_mm in patch_info:
            uv_slice = raw_uv[start:end].copy()

            # Un-flip back patches to restore original parameterisation space
            if was_flipped:
                uv_slice[:, 1] *= -1

            # Scale to mm and center by outer boundary mean (same as Stage2 preprocessing)
            uv_c = uv_slice * 1000.0 - bm_mm

            T = tsol.get(pid, Rigid2D(0.0, 0.0, 0.0))
            uv_t = T.apply(uv_c)

            k = kappas_by_id.get(pid, 0)
            uv_t += (k / K) * np.array([period_u_mm, period_v_mm])

            fabric_uv[start:end] = uv_t

        return fabric_uv

    def save_frame_ply(self):
        V_merged = self.state_0.particle_q.numpy().astype(np.float32) / self.scale
        V_unmerged = V_merged[self.unmerged_to_merged]
        mesh = trimesh.Trimesh(vertices=V_unmerged, faces=self.unmerged_faces, process=False)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        out_path = str(self.out_dir / f"cloth_{self.frame_idx:05d}.ply")
        add_uv_coordinates(trimesh_to_plydata(mesh), self.garment_mesh_uv, out_path)

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            # IMPORTANT: update contacts for the current state each substep
            #self.contacts = self.model.collide(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

        if (self.frame_idx % self.save_every) == 0:
            self.save_frame_ply()

        self.frame_idx += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        pass


def run_simulation(avatar):
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example and run
    example = Example(viewer, avatar)

    newton.examples.run(example, args)
    #newton.examples.run(example)


def run_headless_simulation(avatar, garment_type, is_adapt=False,
                             tsol=None, kappas_by_id=None, K=None,
                             period_u_mm=None, period_v_mm=None,
                             pattern_root='results/pattern/latest',
                             patches_dir='data/patches/upper'):
    viewer = HeadlessViewer()
    example = Example(viewer, avatar, garment_type, is_adapt,
                      tsol=tsol, kappas_by_id=kappas_by_id, K=K,
                      period_u_mm=period_u_mm, period_v_mm=period_v_mm,
                      pattern_root=pattern_root,
                      patches_dir=patches_dir)

    # run headless for some number of frames
    num_frames = 60  # 1 secons @ 60 fps
    for f in range(num_frames):
        example.step()


if __name__ == "__main__":
    avatar_path = 'data/body/ref.ply'
    #avatar_path = 'data/body/target-shape.ply'
    run_simulation(avatar_path)   # open a window for rendering
    #run_headless_simulation()
