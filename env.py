from http import client
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
        max_len = 2 * math.pi
        target = []
        idx = 0
        for i in range(len(self.joint_idx)):
            if self.dof_count[i] == 4:
                exp_map = action[idx:idx+3]
                exp_len = np.linalg.norm(exp_map)   #l2 norm
                if exp_len > max_len:
                    exp_map = exp_map * (max_len / exp_len)
                #convert exp_map to axis angle
                theta = np.linalg.norm(exp_map)
                if theta > 0.000001:
                    axis = exp_map / theta
                    norm_theta = math.fmod(theta, 2*math.pi)
                    if norm_theta > math.pi:
                        norm_theta = -2 * math.pi + norm_theta
                    elif norm_theta < -math.pi:
                        norm_theta = 2 * math.pi + norm_theta
                    theta = norm_theta
                else:
                    axis = np.array([0,0,1])
                    theta = 0
                quat = self.client.getQuaternionFromAxisAngle(axis.tolist(), theta)
                target.append(list(quat))
                idx += 3
            elif self.dof_count[i] == 1:
                target.append([action[idx]])
                idx += 1
        return target

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

class Env():
    def __init__(self, args):
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

        with open(args.motion_file) as motion_file:
            self.mocap = json.load(motion_file)

    def step(self, action):
        self.sim_model.ApplyAction(action)
        self.client.stepSimulation()
        return

if __name__ == '__main__':
    class a():
        draw = True
        timestep = 1/240
        motion_file = "./motion_file/sfu_walking.txt"
        fall_contact_bodies = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14]
    arg = a()

    env = Env(arg)
    pose = {
        'base_pos': env.mocap['Frames'][1][1:4],
        'base_ori': env.mocap['Frames'][1][4:8],
        'base_lin_vel': [0,0,0],
        'base_ang_vel': [0,0,0],
        'joint_pos': env.mocap['Frames'][1][8:],
        'joint_vel': [0]*env.sim_model.total_dof,
    }
    env.sim_model.InitPose(pose)
    for i in range(len(env.mocap['Frames'])):
        action=[]
        dof_idx = 8
        for idx in range(len(env.sim_model.joint_idx)):
            if env.sim_model.dof_count[idx] == 4:
                quat = [
                    env.mocap['Frames'][i][dof_idx + 1],
                    env.mocap['Frames'][i][dof_idx + 2],
                    env.mocap['Frames'][i][dof_idx + 3],
                    env.mocap['Frames'][i][dof_idx + 0],
                ]
                axis, angle = env.client.getAxisAngleFromQuaternion(quat)
                map = [
                    axis[0] * angle,
                    axis[1] * angle,
                    axis[2] * angle,
                ]
            elif env.sim_model.dof_count[idx] == 1:
                map = [env.mocap['Frames'][i][dof_idx]]
            action = action + map
            dof_idx += env.sim_model.dof_count[idx]
        action = np.array(action)
        env.sim_model.ApplyAction(action)
        env.client.stepSimulation()
        time.sleep(1/240)
        env.client.stepSimulation()
        time.sleep(1/240)