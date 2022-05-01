from asyncio import FastChildWatcher
from http import client
import random
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc
from pybullet_utils import pd_controller_stable as pd_controller
import math
import json
import numpy as np

import time

chest = 1
neck = 2
right_hip = 3
right_knee = 4
right_ankle = 5
right_shoulder = 6
right_elbow = 7
left_hip = 9
left_knee = 10
left_ankle = 11
left_shoulder = 12
left_elbow = 13
joint_friction_force = 0

class Humanoid():
    def __init__(self, args, client, sim_model=True):
        self.client = client
        self.timestep = args.timestep
        if sim_model:
            flags = p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        else:
            flags = p.URDF_MAINTAIN_LINK_ORDER
        self.humanoid = self.client.loadURDF("./models/humanoid.urdf",
                                            [0, 0.889540259, 0],
                                            globalScaling=0.25,
                                            useFixedBase=not sim_model,
                                            flags=flags)
        self.client.changeDynamics(self.humanoid, -1, linearDamping=0, angularDamping=0)

        if sim_model:
            self.client.changeDynamics(self.humanoid, -1, lateralFriction=0.9)
            for j in range(self.client.getNumJoints(self.humanoid)):
                self.client.changeDynamics(self.humanoid, j, lateralFriction=0.9)
        else:
            self.client.setCollisionFilterGroupMask(self.humanoid,
                                                    -1,
                                                    collisionFilterGroup=0,
                                                    collisionFilterMask=0)
            self.client.changeDynamics(self.humanoid,
                                        -1,
                                        activationState=p.ACTIVATION_STATE_SLEEP +
                                                        p.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                        p.ACTIVATION_STATE_DISABLE_WAKEUP)
            self.client.changeVisualShape(self.humanoid, -1, rgbaColor=[1, 1, 1, 0.4])
            for j in range(self.client.getNumJoints(self.humanoid)):
                self.client.setCollisionFilterGroupMask(self.humanoid,
                                                        j,
                                                        collisionFilterGroup=0,
                                                        collisionFilterMask=0)
                self.client.changeDynamics(self.humanoid,
                                            j,
                                            activationState=p.ACTIVATION_STATE_SLEEP +
                                                            p.ACTIVATION_STATE_ENABLE_SLEEPING +
                                                            p.ACTIVATION_STATE_DISABLE_WAKEUP)
                self.client.changeVisualShape(self.humanoid, j, rgbaColor=[1, 1, 1, 0.4])

        self.end_effectors = [5, 8, 11, 14]
        self.max_force = [
            200, 200, 200, 200, 50, 50, 50, 50, 200, 200, 200, 200, 150, 90,
            90, 90, 90, 100, 100, 100, 100, 60, 200, 200, 200, 200, 150, 90, 90, 90, 90, 100, 100,
            100, 100, 60
        ]
        self.kp = [
            1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500,
            400, 400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400,
            400, 400, 400, 400, 300
        ]
        self.kd = [
            100, 100, 100, 100, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40,
            40, 40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
        ]
        self.joint_idx = [
            chest, neck, right_hip, right_knee, right_ankle, right_shoulder, right_elbow, left_hip, left_knee,
            left_ankle, left_shoulder, left_elbow
        ]
        self.dof_count = [4, 4, 4, 1, 4, 4, 1, 4, 1, 4, 4, 1]
        self.total_dof = 0
        for dof in self.dof_count:
            self.total_dof += dof

        for j in self.joint_idx:
            self.client.setJointMotorControl2(self.humanoid,
                                              j,
                                              p.POSITION_CONTROL,
                                              targetPosition=0,
                                              positionGain=0,
                                              targetVelocity=0,
                                              force=joint_friction_force)
            self.client.setJointMotorControlMultiDof(self.humanoid,
                                                     j,
                                                     p.POSITION_CONTROL,
                                                     targetPosition=[0, 0, 0, 1],
                                                     targetVelocity=[0, 0, 0],
                                                     positionGain=0,
                                                     velocityGain=1,
                                                     force=[joint_friction_force]*3)

        self.fall_contact_bodies = []
        if args.fall_contact_bodies is not None:
            self.fall_contact_bodies = args.fall_contact_bodies

        self.stable_pd = pd_controller.PDControllerStableMultiDof(self.client)

    def InitPose(self, pose):
        '''
        pose['base_pos']: base_pos
        pose['base_ori']: base_ori
        pose['base_lin_vel']: base linear velocity
        pose['base_ang_vel']: base angular velocity
        pose['joint_pos']: joint position, order check self.joint_idx
        pose['joint_vel']: joint velocity, order check self.joint_idx, there is one zero padding for each spheric joint
        '''
        # TODO: remember we swap order of quaternion here
        pose['base_ori'] = pose['base_ori'][1:] + [pose['base_ori'][0]]
        self.client.resetBasePositionAndOrientation(self.humanoid, pose['base_pos'], pose['base_ori'])
        self.client.resetBaseVelocity(self.humanoid, pose['base_lin_vel'], pose['base_ang_vel'])
        target_pos=[]
        target_vel=[]
        dof_idx = 0
        for idx in range(len(self.joint_idx)):
            if self.dof_count[idx] == 4:
                pos = [
                    pose['joint_pos'][dof_idx + 1],
                    pose['joint_pos'][dof_idx + 2],
                    pose['joint_pos'][dof_idx + 3],
                    pose['joint_pos'][dof_idx + 0],
                ]
                vel = [
                    pose['joint_vel'][dof_idx + 0],
                    pose['joint_vel'][dof_idx + 1],
                    pose['joint_vel'][dof_idx + 2],
                ]
            elif self.dof_count[idx] == 1:
                pos = [pose['joint_pos'][dof_idx]]
                vel = [pose['joint_vel'][dof_idx]]
            target_pos.append(pos)
            target_vel.append(vel)
            dof_idx += self.dof_count[idx]
        
        self.client.resetJointStatesMultiDof(self.humanoid, self.joint_idx,
                                             target_pos, target_vel)

    def ApplyAction(self, action: np.array):
        target = self.ActionToTarget(action)
        self.ComputeAndApplyPDForces(target, self.max_force)

    def ActionToTarget(self, action: np.array):
        target = []
        idx = 0
        for i in range(len(self.joint_idx)):
            if self.dof_count[i] == 4:
                angle = action[idx]
                axis = [action[idx+1], action[idx+2], action[idx+3]]
                rot = self.client.getQuaternionFromAxisAngle(axis, angle)
                target.append(list(rot))
                idx += 4
            elif self.dof_count[i] == 1:
                target.append([action[idx]])
                idx += 1
        return target

        # max_len = 2 * math.pi
        # target = []
        # idx = 0
        # for i in range(len(self.joint_idx)):
        #     if self.dof_count[i] == 4:
        #         exp_map = action[idx:idx+3]
        #         exp_len = np.linalg.norm(exp_map)   #l2 norm
        #         if exp_len > max_len:
        #             exp_map = exp_map * (max_len / exp_len)
        #         #convert exp_map to axis angle
        #         theta = np.linalg.norm(exp_map)
        #         if theta > 0.000001:
        #             axis = exp_map / theta
        #             norm_theta = math.fmod(theta, 2*math.pi)
        #             if norm_theta > math.pi:
        #                 norm_theta = -2 * math.pi + norm_theta
        #             elif norm_theta < -math.pi:
        #                 norm_theta = 2 * math.pi + norm_theta
        #             theta = norm_theta
        #         else:
        #             axis = np.array([0,0,1])
        #             theta = 0
        #         quat = self.client.getQuaternionFromAxisAngle(axis.tolist(), theta)
        #         target.append(list(quat))
        #         idx += 3
        #     elif self.dof_count[i] == 1:
        #         target.append([action[idx]])
        #         idx += 1
        # return target

    def ComputeAndApplyPDForces(self, target, max_force):
        scaling = 1
        forces = []
        target_pos=[]
        target_vel=[]
        kps = []
        kds = []
        dof_idx = 0
        for idx in range(len(self.joint_idx)):
            kps.append(self.kp[dof_idx])
            kds.append(self.kd[dof_idx])
            if self.dof_count[idx] == 4:
                force = [
                    scaling * max_force[dof_idx + 0],
                    scaling * max_force[dof_idx + 1],
                    scaling * max_force[dof_idx + 2],
                ]
                vel = [0, 0, 0]
            elif self.dof_count[idx] == 1:
                force = [scaling * max_force[dof_idx]]
                vel = [0]
            forces.append(force)
            target_vel.append(vel)
            dof_idx += self.dof_count[idx]
        self.client.setJointMotorControlMultiDofArray(self.humanoid,
                                                      self.joint_idx,
                                                      p.STABLE_PD_CONTROL,
                                                      targetPositions = target,
                                                      targetVelocities = target_vel,
                                                      forces = forces,
                                                      positionGains = kps,
                                                      velocityGains = kds,)

    def SampleFrame(self):
        frame = []
        base_pos, base_ori = self.client.getBasePositionAndOrientation(self.humanoid)
        frame += list(base_pos)
        frame += [base_ori[3]] + list(base_ori[:3])
        joints_state = self.client.getJointStatesMultiDof(self.humanoid, self.joint_idx)
        for state in joints_state:
            if len(state[0]) == 4: #spherical joint
                frame += [state[0][3]] + list(state[0][:3])
            else:
                frame += state[0]
        return frame

    def SampleVel(self):
        vel = []
        #calculate base pos and ori first
        base_vel = self.client.getBaseVelocity(self.humanoid)
        vel += list(base_vel[0])
        vel += list(base_vel[1]) + [0]    #add a zero padding here to sync with InitPose
        joints_state = self.client.getJointStatesMultiDof(self.humanoid, self.joint_idx)
        for state in joints_state:
            if len(state[1]) == 3:  #spherical joint
                vel += list(state[1]) + [0]   #add a zero padding here to sync with InitPose
            else:
                vel += list(state[1])
        return vel

    def QuatQuatMult(self, quat1, quat2):
        q1_w = quat1[0]
        q1_x = quat1[1]
        q1_y = quat1[2]
        q1_z = quat1[3]
        q2_w = quat2[0]
        q2_x = quat2[1]
        q2_y = quat2[2]
        q2_z = quat2[3]
        w = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
        x = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
        y = q1_w * q2_y + q1_y * q2_w + q1_z * q2_x - q1_x * q2_z
        z = q1_w * q2_z + q1_z * q2_w + q1_x * q2_y - q1_y * q2_x
        return [w,x,y,z]

    def QuatVecMult(self, quat, vec):
        #use pybullet quaternion order here
        x = vec[0]
        y = vec[1]
        z = vec[2]
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]
        #q*v
        ix =  qw * x + qy * z - qz * y
        iy =  qw * y + qz * x - qx * z
        iz =  qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        target = []
        target.append(ix * qw + iw * -qx + iy * -qz - iz * -qy)
        target.append(iy * qw + iw * -qy + iz * -qx - ix * -qz)
        target.append(iz * qw + iw * -qz + ix * -qy - iy * -qx)
        return target

    def CalNormalTangent(self, q):
        return self.QuatVecMult(q, [0, 1, 0]), self.QuatVecMult(q, [1, 0, 0])
        
    def RecordObs(self):
        state = []

        #build origin trans
        root_pos, root_ori = self.client.getBasePositionAndOrientation(self.humanoid)
        inv_root_pos = [-root_pos[0], 0, -root_pos[2]]
        eul = self.client.getEulerFromQuaternion(root_ori)
        ref_dir = [1, 0, 0]
        rot_vec = self.client.rotateVector(root_ori, ref_dir)
        heading = math.atan2(-rot_vec[2], rot_vec[0])
        heading2 = eul[1]
        heading_ori = self.client.getQuaternionFromAxisAngle([0, 1, 0], -heading)
        heading_mat = self.client.getMatrixFromQuaternion(heading_ori)
        root_trans_pos, root_trans_ori = self.client.multiplyTransforms([0, 0, 0],
                                                                        heading_ori,
                                                                        inv_root_pos,
                                                                        [0, 0, 0, 1])
        base_pos, base_ori = root_pos, root_ori

        root_pos_rel, dummy = self.client.multiplyTransforms(root_trans_pos, root_trans_ori,
                                                                base_pos, [0, 0, 0, 1])
                                                                
        local_pos, local_ori = self.client.multiplyTransforms(root_trans_pos, root_trans_ori,
                                                                base_pos, base_ori)

        local_pos = [
            local_pos[0] - root_pos_rel[0], local_pos[1] - root_pos_rel[1], local_pos[2] - root_pos_rel[2]
        ]

        state.append(root_pos_rel[1])

        self.pb2dmJoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        linkIndicesSim = []
        for pbJoint in range(self.client.getNumJoints(self.humanoid)):
            linkIndicesSim.append(self.pb2dmJoints[pbJoint])
        
        linkStatesSim = self.client.getLinkStates(self.humanoid, linkIndicesSim, computeForwardKinematics=True, computeLinkVelocity=True)
        
        for pbJoint in range(self.client.getNumJoints(self.humanoid)):
            j = self.pb2dmJoints[pbJoint]
            
            ls = linkStatesSim[pbJoint]
            linkPos = ls[0]
            linkOrn = ls[1]
            linkPosLocal, linkOrnLocal = self.client.multiplyTransforms(root_trans_pos, root_trans_ori, linkPos, linkOrn)
            if (linkOrnLocal[3] < 0):
                linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
            linkPosLocal = [
                linkPosLocal[0] - root_pos_rel[0], linkPosLocal[1] - root_pos_rel[1],
                linkPosLocal[2] - root_pos_rel[2]
            ]
            for l in linkPosLocal:
                state.append(l)

            if (linkOrnLocal[3] < 0):
                linkOrnLocal[0] *= -1
                linkOrnLocal[1] *= -1
                linkOrnLocal[2] *= -1
                linkOrnLocal[3] *= -1

            linkOrnLocalNormal, linkOrnLocalTangent = self.CalNormalTangent(linkOrnLocal)

            state += linkOrnLocalNormal
            state += linkOrnLocalTangent
        
        for pbJoint in range(self.client.getNumJoints(self.humanoid)):
            j = self.pb2dmJoints[pbJoint]
            ls = linkStatesSim[pbJoint]
            
            linkLinVel = ls[6]
            linkAngVel = ls[7]
            linkLinVelLocal, unused = self.client.multiplyTransforms([0, 0, 0], root_trans_ori,
                                                                                linkLinVel, [0, 0, 0, 1])
            linkAngVelLocal, unused = self.client.multiplyTransforms([0, 0, 0], root_trans_ori,
                                                                                linkAngVel, [0, 0, 0, 1])

            for l in linkLinVelLocal:
                state.append(l)
            for l in linkAngVelLocal:
                state.append(l)

        return np.array(state)

    def terminates(self):
        #check if any non-allowed body part hits the ground
        terminates = False

        base_pos, base_ori = self.client.getBasePositionAndOrientation(self.humanoid)
        terminates |= base_pos[1] > 5

        if not terminates:
            pts = self.client.getContactPoints()
            for p in pts:
                part = -1
                #ignore self-collision
                if (p[1] == p[2]):
                    continue
                if (p[1] == self.humanoid):
                    part = p[3]
                if (p[2] == self.humanoid):
                    part = p[4]
                if (part >= 0 and part in self.fall_contact_bodies):
                    #print("terminating part:", part)
                    terminates = True

        return terminates

    def had_problem(self):
        had_problem = False
        vel = self.client.getBaseVelocity(self.humanoid)
        lin_vel = np.array(vel[0])
        ang_vel = np.array(vel[1])
        had_problem |= True if np.sum(np.abs(lin_vel) > 100) else False
        had_problem |= True if np.sum(np.abs(ang_vel) > 100) else False

        base_pos, base_ori = self.client.getBasePositionAndOrientation(self.humanoid)
        had_problem |= abs(base_pos[1]) > 5
        had_problem |= abs(base_pos[0]) > 20
        had_problem |= abs(base_pos[2]) > 20

        return had_problem

    def GetStateDim(self):
        state = self.RecordObs()
        return len(state)

    def GetActionDim(self):
        dim = 0
        for i in self.dof_count:
            if i == 4:
                dim += 4#3
            else: 
                dim += 1
        return dim

class MotionCapture():
    def __init__(self, motion_file, kin_model: Humanoid):
        with open(motion_file) as mocap:
            self.mocap = json.load(mocap)
        self.kin_model = kin_model
        self.duration = self.CalDuration()
        self.frame_duration = self.mocap['Frames'][0][0]
        self.num_frames = len(self.mocap['Frames'])

    def CalDuration(self):
        duration = 0
        for frame in self.mocap['Frames']:
            duration += frame[0]
        return duration

    def ComputeAngVel(self, ori_prev, ori_next, dt):
        dorn = self.kin_model.client.getDifferenceQuaternion(ori_prev, ori_next)
        axis, angle = self.kin_model.client.getAxisAngleFromQuaternion(dorn)
        ang_vel = [(axis[0] * angle) / dt, (axis[1] * angle) / dt, (axis[2] * angle) / dt]
        return ang_vel

    def SampleFrame(self, timestamp):
        phase = timestamp / self.duration
        frame_fraction = phase * self.num_frames
        prev_frame_idx = math.floor(frame_fraction)
        next_frame_idx = math.ceil(frame_fraction)
        frame_fraction -= prev_frame_idx
        # print(prev_frame_idx, next_frame_idx, frame_fraction, timestamp, self.duration, self.num_frames)
        prev_frame = np.array(self.mocap['Frames'][prev_frame_idx])
        next_frame = np.array(self.mocap['Frames'][next_frame_idx])
        # frame = prev_frame + frame_fraction * (next_frame - prev_frame)
        # frame = frame.tolist()[1:]
        frame = []
        #calculate base pos and ori first
        frame += [
            prev_frame[1] + frame_fraction * (next_frame[1] - prev_frame[1]),
            prev_frame[2] + frame_fraction * (next_frame[2] - prev_frame[2]),
            prev_frame[3] + frame_fraction * (next_frame[3] - prev_frame[3]),
        ]
        prev_base_ori = [prev_frame[5], prev_frame[6], prev_frame[7], prev_frame[4]]
        next_base_ori = [next_frame[5], next_frame[6], next_frame[7], next_frame[4]]
        base_ori = list(self.kin_model.client.getQuaternionSlerp(prev_base_ori, next_base_ori, frame_fraction))
        frame += [base_ori[3]] + base_ori[:3]
        dof_idx = 8
        for idx in range(len(self.kin_model.joint_idx)):
            if self.kin_model.dof_count[idx] == 4:
                prev_ori = [prev_frame[dof_idx+1], prev_frame[dof_idx+2], prev_frame[dof_idx+3], prev_frame[dof_idx]]
                next_ori = [next_frame[dof_idx+1], next_frame[dof_idx+2], next_frame[dof_idx+3], next_frame[dof_idx]]
                ori = list(self.kin_model.client.getQuaternionSlerp(prev_ori, next_ori, frame_fraction))
                frame += [ori[3]] + ori[:3]
            elif self.kin_model.dof_count[idx] == 1:
                frame += [prev_frame[dof_idx] + frame_fraction * (next_frame[dof_idx] - prev_frame[dof_idx])]
            dof_idx += self.kin_model.dof_count[idx]
        return frame

    def SampleVel(self, timestamp):
        phase = timestamp / self.duration
        prev_frame_idx = math.floor(phase * self.num_frames)
        next_frame_idx = prev_frame_idx + 1
        prev_frame = np.array(self.mocap['Frames'][prev_frame_idx])
        next_frame = np.array(self.mocap['Frames'][next_frame_idx])
        vel = []
        #calculate base pos and ori first
        vel += [
            (next_frame[1] - prev_frame[1]) / self.frame_duration,
            (next_frame[2] - prev_frame[2]) / self.frame_duration,
            (next_frame[3] - prev_frame[3]) / self.frame_duration,
        ]
        prev_base_ori = [prev_frame[5], prev_frame[6], prev_frame[7], prev_frame[4]]
        next_base_ori = [next_frame[5], next_frame[6], next_frame[7], next_frame[4]]
        vel += self.ComputeAngVel(prev_base_ori, next_base_ori, self.frame_duration) + [0] #add a zero padding here to sync with InitPose
        dof_idx = 8
        for idx in range(len(self.kin_model.joint_idx)):
            if self.kin_model.dof_count[idx] == 4:
                prev_ori = [prev_frame[dof_idx+1], prev_frame[dof_idx+2], prev_frame[dof_idx+3], prev_frame[dof_idx]]
                next_ori = [next_frame[dof_idx+1], next_frame[dof_idx+2], next_frame[dof_idx+3], next_frame[dof_idx]]
                vel += self.ComputeAngVel(prev_ori, next_ori, self.frame_duration) + [0] #add a zero padding here to sync with InitPose
            elif self.kin_model.dof_count[idx] == 1:
                vel += [(next_frame[dof_idx] - prev_frame[dof_idx]) / self.frame_duration]
            dof_idx += self.kin_model.dof_count[idx]
        return vel

    def RecordObs(self, timestep):
        prev_frame_time = random.uniform(timestep*4, self.duration-timestep*4) #minius more timestep in case some error
        next_frame_time = prev_frame_time + timestep
        prev_frame = self.SampleFrame(prev_frame_time)
        prev_vel = self.SampleVel(prev_frame_time)
        next_frame = self.SampleFrame(next_frame_time)
        next_vel = self.SampleVel(next_frame_time)

        ground_h = 0

        #calculate the ref_origin_rot
        root_rot = prev_frame[3:7]
        ref_dir = [1, 0 , 0]
        rot_dir = self.QuatVecMult(root_rot, ref_dir)
        heading = math.atan2(-rot_dir[2], rot_dir[0])
        axis = [0, 1, 0]
        ref_origin_rot = self.kin_model.client.getQuaternionFromAxisAngle(axis, -heading)
        ref_origin_rot = [ref_origin_rot[3]] + list(ref_origin_rot[:3])   #reorder according to deepmimic format

        obs = []
        obs += self.RecordObsPose(prev_frame, ground_h, ref_origin_rot)
        obs += self.RecordObsPose(next_frame, ground_h, ref_origin_rot)
        obs += self.RecordObsVel(prev_vel, ref_origin_rot)
        obs += self.RecordObsVel(next_frame, ref_origin_rot)

        return obs

    def RecordObsPose(self, pose, ground_h, ref_origin_rot):
        obs = []
        root_pos = pose[:3]
        root_rot = pose[3:7]

        root_h = root_pos[1] - ground_h
        obs.append(root_h)

        root_rot = self.QuatQuatMult(ref_origin_rot, root_rot)
        root_rot_norm, root_rot_tan = self.CalNormalTangent(root_rot)
        obs += root_rot_norm
        obs += root_rot_tan

        dof_idx = 7
        for idx in range(len(self.kin_model.joint_idx)):
            if self.kin_model.dof_count[idx] == 4:
                joint_rot = pose[dof_idx:dof_idx+4]
                joint_rot_norm, joint_rot_tan = self.CalNormalTangent(joint_rot)
                obs += joint_rot_norm
                obs += joint_rot_tan
            elif self.kin_model.dof_count[idx] == 1:
                obs.append(pose[dof_idx])
            dof_idx += self.kin_model.dof_count[idx]

        kin_pose = {
            'base_pos': pose[:3],
            'base_ori': pose[3:7],
            'base_lin_vel': [0,0,0],
            'base_ang_vel': [0,0,0],
            'joint_pos': pose[7:],
            'joint_vel': [0] * self.kin_model.total_dof,
        }
        self.kin_model.InitPose(kin_pose)
        for link_idx in self.kin_model.end_effectors:
            link_state = self.kin_model.client.getLinkState(self.kin_model.humanoid, link_idx)
            link_pos = link_state[0]
            link_pos = [link_pos[i]-root_pos[i] for i in range(len(link_pos))]
            link_pos = self.QuatVecMult(ref_origin_rot, link_pos)
            obs += link_pos

        return obs

    def RecordObsVel(self, vel, ref_origin_rot):
        obs = []
        root_vel = vel[:3]
        root_ang_vel = vel[3:6]

        root_vel = self.QuatVecMult(ref_origin_rot, root_vel)
        root_ang_vel = self.QuatVecMult(ref_origin_rot, root_ang_vel)
        obs += root_vel
        obs += root_ang_vel

        dof_idx = 7
        for idx in range(len(self.kin_model.joint_idx)):
            if self.kin_model.dof_count[idx] == 4:
                joint_rot = vel[dof_idx:dof_idx+3]
                obs += joint_rot
            elif self.kin_model.dof_count[idx] == 1:
                obs.append(vel[dof_idx])
            dof_idx += self.kin_model.dof_count[idx]

        return obs

    def CalNormalTangent(self, q):
        return self.QuatVecMult(q, [0, 1, 0]), self.QuatVecMult(q, [1, 0, 0])

    def QuatQuatMult(self, quat1, quat2):
        q1_w = quat1[0]
        q1_x = quat1[1]
        q1_y = quat1[2]
        q1_z = quat1[3]
        q2_w = quat2[0]
        q2_x = quat2[1]
        q2_y = quat2[2]
        q2_z = quat2[3]
        w = q1_w * q2_w - q1_x * q2_x - q1_y * q2_y - q1_z * q2_z
        x = q1_w * q2_x + q1_x * q2_w + q1_y * q2_z - q1_z * q2_y
        y = q1_w * q2_y + q1_y * q2_w + q1_z * q2_x - q1_x * q2_z
        z = q1_w * q2_z + q1_z * q2_w + q1_x * q2_y - q1_y * q2_x
        return [w,x,y,z]

    def QuatVecMult(self, quat, vec):
        x = vec[0]
        y = vec[1]
        z = vec[2]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]
        qw = quat[0]
        #q*v
        ix =  qw * x + qy * z - qz * y
        iy =  qw * y + qz * x - qx * z
        iz =  qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        target = []
        target.append(ix * qw + iw * -qx + iy * -qz - iz * -qy)
        target.append(iy * qw + iw * -qy + iz * -qx - ix * -qz)
        target.append(iz * qw + iw * -qz + ix * -qy - iy * -qx)
        return target

    def Replay(self):
        steps = math.floor((self.duration-self.kin_model.timestep) / self.kin_model.timestep)
        for i in range(steps):
            pos = self.SampleFrame(i*self.kin_model.timestep)
            vel = self.SampleVel(i*self.kin_model.timestep)
            pose = {
                'base_pos': pos[:3],
                'base_ori': pos[3:7],
                'base_lin_vel': vel[:3],
                'base_ang_vel': vel[3:6],
                'joint_pos': pos[7:],
                'joint_vel': vel[6:],
            }
            self.kin_model.InitPose(pose)
            self.kin_model.client.stepSimulation()
            time.sleep(self.kin_model.timestep)

class Env():
    def __init__(self, args):
        self.draw = args.draw
        if args.draw:
            self.client = bc.BulletClient(p.GUI)
            self.client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = bc.BulletClient(p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
        self.client.setGravity(0,-9.81,0)
        self.client.setPhysicsEngineParameter(numSolverIterations=10)
        self.client.setPhysicsEngineParameter(numSubSteps=1)
        self.timestep = args.timestep
        self.client.setTimeStep(self.timestep)

        self.plane = self.client.loadURDF("plane_implicit.urdf",
                                        [0,0,0], 
                                        self.client.getQuaternionFromEuler([-math.pi/2, 0, 0]),
                                        useMaximalCoordinates=True)
        self.client.changeDynamics(self.plane, -1, lateralFriction=0.9)

        self.sim_model = Humanoid(args, self.client, sim_model=True)
        self.kin_model = Humanoid(args, self.client, sim_model=False)

        self.mocap = MotionCapture(args.motion_file, self.kin_model)

        self.sim_model_prev_frame = None
        self.sim_model_curr_frame = None

    def reset(self):
        frame_time = random.uniform(self.timestep, self.mocap.duration-self.timestep*2)
        frame = self.mocap.SampleFrame(frame_time)
        frame_vel = self.mocap.SampleVel(frame_time)
        pose = {
            'base_pos': frame[:3],
            'base_ori': frame[3:7],
            'base_lin_vel': frame_vel[:3],
            'base_ang_vel': frame_vel[3:6],
            'joint_pos': frame[7:],
            'joint_vel': [0]*self.sim_model.total_dof,#frame_vel[6:],
        }
        self.sim_model.InitPose(pose)

    def step(self, action):
        frame, vel = self.RecordAgentFrameAndVel()
        self.sim_model_prev_frame = [frame, vel]
        self.sim_model.ApplyAction(action)
        self.client.stepSimulation()
        frame, vel = self.RecordAgentFrameAndVel()
        self.sim_model_curr_frame = [frame, vel]
        return

    def RecordAgentObs(self):
        agent_obs = self.sim_model.RecordObs()
        return agent_obs

    def RecordAgentFrameAndVel(self):
        frame = self.sim_model.SampleFrame()
        vel = self.sim_model.SampleVel()
        return frame, vel

    def RecordAgentDiscObs(self, prev_frame, prev_vel, next_frame, next_vel):
        ground_h = 0

        #calculate the ref_origin_rot
        root_rot = prev_frame[3:7]
        ref_dir = [1, 0 , 0]
        rot_dir = self.mocap.QuatVecMult(root_rot, ref_dir)
        heading = math.atan2(-rot_dir[2], rot_dir[0])
        axis = [0, 1, 0]
        ref_origin_rot = self.client.getQuaternionFromAxisAngle(axis, -heading)
        ref_origin_rot = [ref_origin_rot[3]] + list(ref_origin_rot[:3])   #reorder according to deepmimic format

        obs = []
        obs += self.mocap.RecordObsPose(prev_frame, ground_h, ref_origin_rot)
        obs += self.mocap.RecordObsPose(next_frame, ground_h, ref_origin_rot)
        obs += self.mocap.RecordObsVel(prev_vel, ref_origin_rot)
        obs += self.mocap.RecordObsVel(next_vel, ref_origin_rot)

        return obs

    def RecordExpertDiscObs(self):
        expert_obs = self.mocap.RecordObs(self.timestep)
        return expert_obs

    def terminate(self):
        terminate = self.sim_model.terminates()
        return terminate

    def GetStateDim(self):
        return self.sim_model.GetStateDim()

    def GetActionDim(self):
        return self.sim_model.GetActionDim()

    def GetDiscInputDim(self):
        expert_obs = self.RecordExpertDiscObs()
        return len(expert_obs)

    def record_amp_obs_expert(self, id):
        return self.RecordExpertDiscObs()

    def record_amp_obs_agent(self, id):
        prev_frame, prev_vel = self.sim_model_prev_frame
        curr_frame, curr_vel = self.sim_model_curr_frame
        return self.RecordAgentDiscObs(prev_frame, prev_vel, curr_frame, curr_vel)

    def build_amp_obs_offset(self, id):
        out_offset = [0] * self.GetDiscInputDim()
        return np.array(out_offset)

    def build_amp_obs_norm_groups(self, id):
        groups = [0] * self.GetDiscInputDim()
        groups[0] = -1
        return groups

    def build_amp_obs_scale(self, agent_id):
        out_scale = [1] * self.GetDiscInputDim()
        return np.array(out_scale)

    def build_state_offset(self, id):
        out_offset = [0] * self.GetStateDim()
        return np.array(out_offset)

    def build_state_norm_groups(self, id):
        groups = [0] * self.GetStateDim()
        groups[0] = -1
        return groups

    def build_state_scale(self, agent_id):
        out_scale = [1] * self.GetStateDim()
        return np.array(out_scale)

    def build_goal_norm_groups(self, agent_id):
        return np.array([])

    def build_goal_offset(self, agent_id):
        return np.array([])

    def build_goal_scale(self, agent_id):
        return np.array([])

    def build_action_offset(self, agent_id):
        # out_offset = [0] * self.GetActionDim()
        out_offset = [
            0.0000000000, 0.0000000000, 0.0000000000, -0.200000000, 0.0000000000, 0.0000000000,
            0.0000000000, -0.200000000, 0.0000000000, 0.0000000000, 0.00000000, -0.2000000, 1.57000000,
            0.00000000, 0.00000000, 0.00000000, -0.2000000, 0.00000000, 0.00000000, 0.00000000,
            -0.2000000, -1.5700000, 0.00000000, 0.00000000, 0.00000000, -0.2000000, 1.57000000,
            0.00000000, 0.00000000, 0.00000000, -0.2000000, 0.00000000, 0.00000000, 0.00000000,
            -0.2000000, -1.5700000
        ]
        return np.array(out_offset)

    def build_action_scale(self, agent_id):
        # out_scale = [1] * self.GetActionDim()
        #see cCtCtrlUtil::BuildOffsetScalePDPrismatic and
        #see cCtCtrlUtil::BuildOffsetScalePDSpherical
        out_scale = [
            0.20833333333333, 1.00000000000000, 1.00000000000000, 1.00000000000000, 0.25000000000000,
            1.00000000000000, 1.00000000000000, 1.00000000000000, 0.12077294685990, 1.00000000000000,
            1.000000000000, 1.000000000000, 0.159235668789, 0.159235668789, 1.000000000000,
            1.000000000000, 1.000000000000, 0.079617834394, 1.000000000000, 1.000000000000,
            1.000000000000, 0.159235668789, 0.120772946859, 1.000000000000, 1.000000000000,
            1.000000000000, 0.159235668789, 0.159235668789, 1.000000000000, 1.000000000000,
            1.000000000000, 0.107758620689, 1.000000000000, 1.000000000000, 1.000000000000,
            0.159235668789
        ]
        return np.array(out_scale)

    def build_action_bound_min(self, agent_id):
        #see cCtCtrlUtil::BuildBoundsPDSpherical
        # out_scale = [-1] * self.get_action_size(agent_id)
        out_scale = [
            -4.79999999999, -1.00000000000, -1.00000000000, -1.00000000000, -4.00000000000,
            -1.00000000000, -1.00000000000, -1.00000000000, -7.77999999999, -1.00000000000,
            -1.000000000, -1.000000000, -7.850000000, -6.280000000, -1.000000000, -1.000000000,
            -1.000000000, -12.56000000, -1.000000000, -1.000000000, -1.000000000, -4.710000000,
            -7.779999999, -1.000000000, -1.000000000, -1.000000000, -7.850000000, -6.280000000,
            -1.000000000, -1.000000000, -1.000000000, -8.460000000, -1.000000000, -1.000000000,
            -1.000000000, -4.710000000
        ]

        return out_scale

    def build_action_bound_max(self, agent_id):
        # out_scale = [1] * self.get_action_size(agent_id)
        out_scale = [
            4.799999999, 1.000000000, 1.000000000, 1.000000000, 4.000000000, 1.000000000, 1.000000000,
            1.000000000, 8.779999999, 1.000000000, 1.0000000, 1.0000000, 4.7100000, 6.2800000,
            1.0000000, 1.0000000, 1.0000000, 12.560000, 1.0000000, 1.0000000, 1.0000000, 7.8500000,
            8.7799999, 1.0000000, 1.0000000, 1.0000000, 4.7100000, 6.2800000, 1.0000000, 1.0000000,
            1.0000000, 10.100000, 1.0000000, 1.0000000, 1.0000000, 7.8500000
        ]
        return out_scale

if __name__ == '__main__':
    class a():
        draw = True
        timestep = 1/240
        motion_file = "./motion_file/sfu_walking.txt"
        fall_contact_bodies = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14]
    arg = a()

    env = Env(arg)
    env.mocap.Replay()
    while True:
        pass
    # for i in range(len(env.mocap.mocap['Frames'])):
    #     pose = {
    #         'base_pos': env.mocap.mocap['Frames'][i][1:4],
    #         'base_ori': env.mocap.mocap['Frames'][i][4:8],
    #         'base_lin_vel': [0,0,0],
    #         'base_ang_vel': [0,0,0],
    #         'joint_pos': env.mocap.mocap['Frames'][i][8:],
    #         'joint_vel': [0]*env.kin_model.total_dof,
    #     }
    #     env.kin_model.InitPose(pose)
    #     env.client.stepSimulation()
    #     time.sleep(1/240)
    #     env.client.stepSimulation()
    #     time.sleep(1/240)
    # pose = {
    #     'base_pos': env.mocap['Frames'][1][1:4],
    #     'base_ori': env.mocap['Frames'][1][4:8],
    #     'base_lin_vel': [0,0,0],
    #     'base_ang_vel': [0,0,0],
    #     'joint_pos': env.mocap['Frames'][1][8:],
    #     'joint_vel': [0]*env.sim_model.total_dof,
    # }
    # env.sim_model.InitPose(pose)
    # for i in range(len(env.mocap['Frames'])):
    #     action=[]
    #     dof_idx = 8
    #     for idx in range(len(env.sim_model.joint_idx)):
    #         if env.sim_model.dof_count[idx] == 4:
    #             quat = [
    #                 env.mocap['Frames'][i][dof_idx + 1],
    #                 env.mocap['Frames'][i][dof_idx + 2],
    #                 env.mocap['Frames'][i][dof_idx + 3],
    #                 env.mocap['Frames'][i][dof_idx + 0],
    #             ]
    #             axis, angle = env.client.getAxisAngleFromQuaternion(quat)
    #             map = [
    #                 axis[0] * angle,
    #                 axis[1] * angle,
    #                 axis[2] * angle,
    #             ]
    #         elif env.sim_model.dof_count[idx] == 1:
    #             map = [env.mocap['Frames'][i][dof_idx]]
    #         action = action + map
    #         dof_idx += env.sim_model.dof_count[idx]
    #     action = np.array(action)
    #     env.sim_model.ApplyAction(action)
    #     env.client.stepSimulation()
    #     time.sleep(1/240)
    #     env.client.stepSimulation()
    #     time.sleep(1/240)