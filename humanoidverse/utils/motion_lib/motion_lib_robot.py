from humanoidverse.utils.motion_lib.motion_lib_base import MotionLibBase
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from .motion_lib_base import MotionlibMode
from humanoidverse.utils.motion_lib.motion_utils.flags import flags

import torch
import numpy as np
import os.path as osp
import glob
from loguru import logger
from rich.progress import track

class MotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        return
    
class LocoMujocoMotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device, simulator):
        super().__init__(motion_lib_cfg = motion_lib_cfg, num_envs = num_envs, device = device)
        self.simulator = simulator
        return

    def load_data(self, motion_file, min_length=-1, im_eval = False):
        '''
            Modify to load motion from numpy files or directories containing pickle files.
            _motion_data_load is like a dictionary with keys as motion names and values as the motion data.
            includes:
            - qpos: T x J, Position of the joints, including the root joint.
            - qvel: T x J, Velocity of the joints, including the root joint.
            - xpos: T x B x 3, Position of all bodies in global coordinates.
            - xquat: T x B x 4, Quaternion of all bodies in global coordinates.
            - cvel: T x B x 6, Velocity of all bodies in global coordinates, v + w.
            - site_xpos: T x S x 3, Position of all sites in global coordinates, mimic.
            - site_xquat: T x S x 4, Quaternion of all sites in global coordinates, mimic.
            - joint_names: List of joint names.
            - body_names: List of body names.
            - site_names: List of site names.
            - frequency: Frequency of the motion data.
            - njnts: Number of joints.
            - split_points: List of split points for the motion data.
        '''

        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = np.load(motion_file, allow_pickle=True)
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(osp.join(motion_file, "*.npz"))
        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
    
            # self._motion_data_list = np.array(list(data_list.values()))
            # self._motion_data_keys = np.array(list(data_list.keys()))
            # self._motion_data_list = list(data_list.values())
            # self._motion_data_keys = list(data_list.keys())
        else:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        # self._num_unique_motions = len(self._motion_data_list)
        self._num_unique_motions = 1
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = np.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def load_motions(self, 
                     random_sample=True, 
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None):
        # import ipdb; ipdb.set_trace()

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False
        _motion_actions = []
        
        if flags.real_traj:
            self.q_gts, self.q_grs, self.q_gavs, self.q_gvs = [], [], [], []

        total_len = 0.0

        # self.num_joints = len(self.skeleton_tree.node_names) #! the old one
        self.num_joints = self._motion_data_load['joint_names'].shape[0] - 1  # -1 for the root joint
        num_motion_to_load = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(num_motion_to_load) + start_idx, self._num_unique_motions ).to(self._device)

        self._curr_motion_ids = sample_idxes
        # self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:5]}, ....")

        for f in track(range(len(sample_idxes)), description="Loading motions..."):
           
            data = self._motion_data_load
            motion_fps = data['frequency'].item()
            curr_dt = 1.0 / motion_fps
            num_frames = data["qpos"].shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
            _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(data)
            _motion_lengths.append(curr_len)
                
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        # import ipdb; ipdb.set_trace()
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)
        
        motion_body_names = self._motion_data_load["body_names"].tolist()
        motion_body_names.remove("world")  # Remove the world body
        root_body_index = motion_body_names.index("root")
        motion_body_names[root_body_index] = "Pelvis"  # Replace the root body with the Pelvis body
        simulator_body_names = self.simulator.body_names
        body_ids = [motion_body_names.index(name) for name in simulator_body_names]
        self.gts = torch.from_numpy(np.concatenate([m["xpos"] for m in motions], axis=0)).float().to(self._device)
        self.gts = self.gts[:, 1:, :]  # Exclude the world body position
        self.gts = self.gts[:, body_ids, :]  # Reorder the indexes to match the simulator body names
        self.grs = torch.from_numpy(np.concatenate([m["xquat"] for m in motions], axis=0)).float().to(self._device)
        self.grs = self.grs[:, 1:, :]  # Exclude the world body quaternion
        self.grs = self.grs[:, body_ids, :]  # Reorder the indexes to match the simulator body names
        self.grs = self.grs[:, :, [1, 2, 3, 0]]  # Reorder the quaternion to match the simulator body names

        root_body_index = self._motion_data_load["body_names"].tolist().index("root")
        self.grvs = torch.from_numpy(np.concatenate([m["cvel"][:, root_body_index, :3] for m in motions], axis=0)).float().to(self._device)
        self.gravs = torch.from_numpy(np.concatenate([m["cvel"][:, root_body_index, 3:] for m in motions], axis=0)).float().to(self._device)

        self.gavs = torch.from_numpy(np.concatenate([m["cvel"][:, :, 3:] for m in motions], axis=0)).float().to(self._device)
        self.gavs = self.gavs[:, 1:, :]  # Exclude the world joint position
        self.gavs = self.gavs[:, body_ids, :]  # Reorder the indexes to match the simulator body names
        self.gvs = torch.from_numpy(np.concatenate([m["cvel"][:, :, :3] for m in motions], axis=0)).float().to(self._device)
        self.gvs = self.gvs[:, 1:, :]  # Exclude the world joint angular velocity
        self.gvs = self.gvs[:, body_ids, :]  # Reorder the indexes to match the simulator body names
        
        motion_joint_names = self._motion_data_load["joint_names"].tolist()[1:]  # Exclude the root joint
        simulator_joint_names = self.simulator.dof_names
        joint_ids = [motion_joint_names.index(name) for name in simulator_joint_names]
        self.dof_pos = torch.from_numpy(np.concatenate([m["qpos"] for m in motions], axis=0)).float().to(self._device)
        self.dof_pos = self.dof_pos[:, 7:]  # Exclude the root joint position
        self.dof_pos = self.dof_pos[:, joint_ids]
        self.dvs = torch.from_numpy(np.concatenate([m["qvel"] for m in motions], axis=0)).float().to(self._device)
        self.dvs = self.dvs[:, 6:]  # Exclude the root joint velocity
        self.dvs = self.dvs[:, joint_ids]

        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = self.gts.clone()
            self.grs_t = self.grs.clone()
            self.gvs_t = self.gvs.clone()
            self.gavs_t = self.gavs.clone()


        # import ipdb; ipdb.set_trace()
        if flags.real_traj:
            self.q_gts = torch.cat(self.q_gts, dim=0).float().to(self._device)
            self.q_grs = torch.cat(self.q_grs, dim=0).float().to(self._device)
            self.q_gavs = torch.cat(self.q_gavs, dim=0).float().to(self._device)
            self.q_gvs = torch.cat(self.q_gvs, dim=0).float().to(self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = self.num_joints
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        return motions