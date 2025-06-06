# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg


import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##

# use local assets
from robot_lab.assets.self_dog import SELF_DOG_CFG  # isort: skip



@configclass
class SelfDogActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )



@configclass
class SelfDogRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    # 跳跃奖励
    lin_vel_z_up_l2 = RewTerm(func=mdp.lin_vel_z_up_l2, weight=0.0)

    lin_acc_z_up_l2 = RewTerm(
        func=mdp.lin_acc_z_up_l2, 
        weight=0.0,
        params={
            "max_angle": 30.0    # 最大姿态角限制(度)
        },
    )

    lin_acc_z_air_l2 = RewTerm(
        func=mdp.lin_acc_z_air_l2, 
        weight=0.0,
        params={
            "max_angle": 30.0    # 最大姿态角限制(度)
        },
    )




    # 奖励配置项
    airborne_reward = RewTerm(
        func=mdp.airborne,
        weight=0.0,  # 腾空奖励权重建议高于接触奖励
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "min_height": 0.3,  # 最低腾空高度阈值
            "max_angle": 30.0    # 最大姿态角限制(度)
        },
    )








@configclass
class SelfDogJumpEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: SelfDogActionsCfg = SelfDogActionsCfg()
    rewards: SelfDogRewardsCfg = SelfDogRewardsCfg()
    
    base_link_name = "base"
    foot_link_name = ".*_wheel"
    wheel_joint_name = ".*_wheel_joint"

    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
    ]



    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = SELF_DOG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="plane",
        # )

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[self.wheel_joint_name]
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[self.wheel_joint_name]
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names


        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names[:-4]
        self.actions.joint_vel.joint_names = self.joint_names[-4:]


        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
         # 随机化轮子的质量
        self.events.randomize_rigid_wheel_mass.params["asset_cfg"].body_names = [self.foot_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-30.0, 30.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-10.0, 10.0)


        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # 设置终止状态的权重为0，表示不考虑终止状态对奖励的影响
        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.0

        self.rewards.flat_orientation_l2.weight = 0

        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 1.0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        

        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]

        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]

        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        self.rewards.joint_acc_wheel_l2.weight =0.0
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])

        self.rewards.joint_power.weight = 0.0
        self.rewards.joint_power.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        self.rewards.joint_mirror.weight = -0.02
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        

        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]


       # Action penalties
        self.rewards.action_rate_l2.weight = 0.0

        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        self.rewards.joint_power.weight = -1e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        self.rewards.stand_still_without_cmd.weight = 0.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]

        # Contact sensor
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]

         # 跳跃任务的奖励，站立和悬空的基础奖励，悬空且达到高度就有奖励
        self.rewards.airborne_reward.weight = 10.0
        self.rewards.airborne_reward.params["sensor_cfg"].body_names = [self.foot_link_name] 
        self.rewards.airborne_reward.params["min_height"] = 0.3
        self.rewards.airborne_reward.params["max_angle"] = 30.0 
        # 机体竖直奖励
        self.rewards.upward.weight = 2.0
        # 在空中加速度奖励
        self.rewards.lin_acc_z_air_l2.weight = 10.0
        # 给予向上速度的奖励
        self.rewards.lin_vel_z_up_l2.weight = 4.0
     


        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.track_ang_vel_z_exp.weight = 4.0


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "SelfDogJumpEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base"]
        self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
