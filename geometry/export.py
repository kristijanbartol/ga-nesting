import os
import numpy as np
import shutil


def export_patches(patches, target_patches_list, valid_patch_idxs, garment_part):
    part_patches_dir = f'data/patches/{garment_part}/'
    if os.path.isdir(part_patches_dir):
        shutil.rmtree(part_patches_dir)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            patch_dir = f'{part_patches_dir}/patch_{patch_idx:02d}/'
            os.makedirs(patch_dir)
            for ext in ['ply', 'obj']:  # need OBJ for the flattening optimization
                patch.export(f'{patch_dir}/ref.{ext}')

                for target_idx in range(len(target_patches_list)):
                    target_patches_list[target_idx][patch_idx].export(f'{patch_dir}/target-{target_idx}.{ext}')


def export_seamlines(seamlines_dict_list, symmetric_seamline_flags, garment_part):
    seamline_dir = f'data/seamlines/{garment_part}/'
    if os.path.isdir(seamline_dir):
        shutil.rmtree(seamline_dir)
    os.makedirs(seamline_dir)
    for seamline_idx, seamline_dict in enumerate(seamlines_dict_list):
        for patch_pair in seamline_dict:
            fpath = f'{seamline_dir}/seam-{seamline_idx}_{patch_pair[0]}-{patch_pair[1]}.txt'
            with open(fpath, mode='w') as seam_f:
                seam_f.write('1\n' if symmetric_seamline_flags[seamline_idx] else '0\n')
                seam_f.write(f'{patch_pair[0]}\n{patch_pair[1]}\n')
                for vidx_pair in seamline_dict[patch_pair]:
                    seam_f.write(f'{vidx_pair[0]} {vidx_pair[1]}\n')


def prepare_scales(body_mesh, patch, is_skirtified):
    scale = 1.0
    scales_u = np.ones(patch.faces.shape[0]) * scale
    scales_v = np.ones(patch.faces.shape[0])

    #ref_kpts = REF_KPTS_SKIRTIFIED if is_skirtified else REF_KPTS
    ref_kpts = REF_KPTS
    top_y = body_mesh.vertices[ref_kpts['lower']['side'][0]][1]
    bottom_y = np.min(patch.vertices[:, 1])
    base_stretch = scale
        
    mean_face_coords = np.mean(patch.vertices[patch.faces], axis=1)
    ref_mask = mean_face_coords[:, 1] < top_y
    scales_u[np.where(ref_mask)] = base_stretch + ((mean_face_coords[ref_mask][:, 1] - top_y) / (bottom_y - top_y)) * (max_stretch - base_stretch)

    return scales_u, scales_v


def export_scales(body_mesh, patches, valid_patch_idxs, garment_part, is_skirtified, max_scale=None):
    part_scales_dir = f'data/scales/{garment_part}/'
    if os.path.isdir(part_scales_dir):
        shutil.rmtree(part_scales_dir)
    for patch_idx, patch in enumerate(patches):
        if patch_idx in valid_patch_idxs:
            patch_scales_dir = os.path.join(part_scales_dir, f'patch_{patch_idx:02d}')
            os.makedirs(patch_scales_dir)

            fpath_u = f'{patch_scales_dir}/scales_u.txt'
            fpath_v = f'{patch_scales_dir}/scales_v.txt'

            scales_u = np.ones(patch.faces.shape[0])
            scales_v = np.ones(patch.faces.shape[0])

            with open(fpath_u, 'w') as f_u:
                for s_u in scales_u:
                    f_u.write(f"{s_u}\n")
            with open(fpath_v, 'w') as f_v:
                for s_v in scales_v:
                    f_v.write(f"{s_v}\n")


def export_patch_labels(patch_labels_dict, garment_part):
    part_labels_dir = f'data/labels/{garment_part}/'
    if os.path.isdir(part_labels_dir):
        shutil.rmtree(part_labels_dir)
    os.makedirs(part_labels_dir)
    for label in patch_labels_dict:
        fpath = os.path.join(part_labels_dir, f'{label}.txt')
        with open(fpath, 'w') as f:
            for patch_idx in patch_labels_dict[label]:
                f.write(f'{patch_idx} ')


def create_latest_dir(valid_patch_idxs, garment_part):
    latest_pattern_result_dir = f'results/pattern/latest/{garment_part}/'
    if os.path.isdir(latest_pattern_result_dir):
        shutil.rmtree(latest_pattern_result_dir)
    for patch_idx in valid_patch_idxs:
        os.makedirs(os.path.join(latest_pattern_result_dir, f'patch_{patch_idx:02d}'))
    scales_dir = f'results/scales/{garment_part}/'
    if os.path.exists(scales_dir):
        shutil.rmtree(scales_dir)
    os.makedirs(scales_dir)


def export_data(patches, valid_patch_idxs, garment_part, seamlines_dict_list, symmetric_seamline_flags, patch_labels_dict, mesh):
    export_patches(patches, [], valid_patch_idxs, garment_part)
    export_seamlines(seamlines_dict_list, symmetric_seamline_flags, garment_part)
    export_scales(mesh, patches, valid_patch_idxs, garment_part, is_skirtified=False)
    export_patch_labels(patch_labels_dict, garment_part)
    create_latest_dir(valid_patch_idxs, garment_part)
