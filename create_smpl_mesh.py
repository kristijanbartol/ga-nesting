from smplx import SMPL
import torch
import numpy as np
import os
import trimesh


if __name__ == "__main__":
    SMPL_DIR = '/Users/kristijanbartol/data/smpl/models/'
    smpl_model = SMPL(
            model_path=os.path.join(SMPL_DIR, f'SMPL_FEMALE.pkl'), 
            gender='female'
        )
    pose = torch.zeros((1, 23 * 3))
    pose[0, 0*3:1*3] = torch.tensor([0, 0, np.pi / 16])
    pose[0, 1*3:2*3] = torch.tensor([0, 0, -np.pi / 16])
    posed_verts = smpl_model(
        betas=torch.zeros((1, 10)),
        body_pose=pose
    ).vertices[0].cpu().detach().numpy()
    
    trimesh.Trimesh(vertices=posed_verts, faces=smpl_model.faces).export('data/SMPL_FEMALE_POSED.ply')
