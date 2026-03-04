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


PANTS = False
HORSE = False


def read_patches(is_adapt=False):
    patches_dir = f'data/patches/upper/'
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
    return trimesh.Trimesh(vertices=merged_mesh.vertices, faces=merged_mesh.faces, process=True)


def read_back_idxs():
    with open('data/labels/upper/back.txt', 'r') as labels_f:
        back_idxs = list(map(int, labels_f.read().split()))
    return back_idxs


def read_sewing_pattern():
    param_2d_dir = os.path.join(f'results/pattern/latest/upper/')
    back_idxs = read_back_idxs()
    patch_2d_meshes = []
    for patch_dirname in sorted(os.listdir(param_2d_dir)):
        patch_dir = f'{param_2d_dir}/{patch_dirname}/'
        patch_idx = int(patch_dirname[-2:])
        param_2d_mesh = trimesh.load(os.path.join(patch_dir, 'optim_final-seams.ply'))
        if patch_idx in back_idxs:
            param_2d_mesh.vertices[:, 1] *= -1
        param_2d_mesh = param_2d_mesh.subdivide()
        patch_2d_meshes.append(param_2d_mesh)
    merged_mesh = trimesh.util.concatenate(patch_2d_meshes)
    return trimesh.Trimesh(vertices=merged_mesh.vertices, faces=merged_mesh.faces, process=False)


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

    def __init__(self, viewer, avatar, is_adapt=False):
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

        garment_mesh_3d = read_patches(is_adapt)
        garment_mesh_2d = read_sewing_pattern()

        if PANTS:
            rim_idxs = extract_upper_rim(garment_mesh_3d.vertices, garment_mesh_3d.faces)

        assert len(garment_mesh_3d.faces) == len(garment_mesh_2d.faces)

        self.cloth_faces = garment_mesh_3d.faces.copy()   # (F, 3) int
        self.save_every = 10
        self.frame_idx = 0
        self.out_dir = Path("./results/simulation/upper/")
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

    def save_frame_ply(self):
        V = self.state_0.particle_q.numpy().astype(np.float32) / self.scale     # scale back to original size
        mesh = trimesh.Trimesh(vertices=V, faces=self.cloth_faces, process=False)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))    # rotate back to original ("laying down")
        mesh.export(self.out_dir / f"cloth_{self.frame_idx:05d}.ply")

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


def run_headless_simulation(avatar, is_adapt=False):
    viewer = HeadlessViewer()
    example = Example(viewer, avatar, is_adapt)

    # run headless for some number of frames
    num_frames = 60  # 1 secons @ 60 fps
    for f in range(num_frames):
        example.step()


if __name__ == "__main__":
    avatar_path = 'data/body/ref.ply'
    #avatar_path = 'data/body/target-shape.ply'
    run_simulation(avatar_path)   # open a window for rendering
    #run_headless_simulation()
