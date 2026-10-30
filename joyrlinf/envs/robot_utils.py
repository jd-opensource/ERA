"""Robot / control-mode helpers for ManiSkill environments."""

from __future__ import annotations


def get_robot_control_mode(policy_setup: str) -> str:
    """Map ``policy_setup`` to ManiSkill ``control_mode`` (aligned with RLinf)."""
    robot = policy_setup
    if robot == "panda-qpos":
        return "pd_joint_delta_pos"
    if robot == "panda-ee-dpos":
        return "pd_ee_delta_pos"
    if robot == "panda-ee-dpose":
        return "pd_ee_delta_pose"
    if robot == "panda-ee-target-dpos":
        return "pd_ee_target_delta_pose"
    if "google_robot_static" in robot:
        return (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_"
            "gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
    if "widowx" in robot:
        return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
    if robot == "panda_wristcam":
        return "pd_ee_body_target_delta_pose_real_root_frame"
    if "panda" in robot:
        return "pd_ee_body_target_delta_pose_real_root_frame"
    raise NotImplementedError(f"Robot policy_setup {policy_setup!r} not supported")
