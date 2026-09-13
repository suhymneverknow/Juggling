"""JAX reward for Mount1 board balancing."""

import jax
import jax.numpy as jnp


REWARD_VERSION = "mount1_jax_board_balance"

ALIVE_REWARD_PER_STEP = 0.005
ACTION_RATE_SCALE = 0.30
BOARD_ANGULAR_SPEED_SCALE = 0.50  # rad/s
BOARD_ANGULAR_SPEED_NEAR_LEVEL_ANGLE = jnp.deg2rad(5.0)
BOARD_ANGULAR_SPEED_TRANSITION_WIDTH = jnp.deg2rad(0.75)
BOARD_ANGULAR_SPEED_FAR_WEIGHT = 0.05
BOARD_ANGULAR_SPEED_NEAR_WEIGHT = 4.00
BOARD_ANGULAR_SPEED_SQRT_EPSILON = 0.01
COM_PIVOT_HORIZONTAL_SCALE = 0.10  # m
FOOT_SLIP_SCALE = 0.20
FOOT_TARGET_XY_SCALE = 0.08  # m
FOOT_TARGET_Z_SCALE = 0.04  # m
JOINT_VELOCITY_SCALE = 1.5
TORSO_Y_COS_SCALE = 0.35
TORSO_Z_SIN_SCALE = 0.35

# The board starts at 15 degrees.  Reward only progress from that reset toward
# level, and make an equal angular improvement increasingly valuable near zero.
BOARD_INITIAL_TILT = jnp.deg2rad(15.0)
BOARD_TILT_WEIGHT = 4.0
BOARD_TILT_CURVATURE = 0.75

# Penalize changes relative to each knee's asymmetric reset IK angle.  The
# first 30 degrees are free; beyond that, the unbounded squared excess makes
# progressively larger deviations increasingly costly.
KNEE_CHANGE_FREE_ANGLE = jnp.deg2rad(30.0)
KNEE_CHANGE_SCALE = jnp.deg2rad(30.0)
KNEE_CHANGE_WEIGHT = 2.0

# Dedicated waist terms avoid dilution by the other twelve controlled joints.
TORSO_JOINT_VELOCITY_SCALE = 0.50  # rad/s
TORSO_ACTION_RATE_SCALE = 0.10
TORSO_ACTION_MAGNITUDE_SCALE = 0.35
TORSO_JOINT_VELOCITY_WEIGHT = 0.60
TORSO_ACTION_RATE_WEIGHT = 0.50
TORSO_ACTION_MAGNITUDE_WEIGHT = 0.30


def _quality(error, scale):
    return jnp.exp(-jnp.square(jnp.clip(error / scale, 0.0, 20.0)))


def mount1_reward(env, state, action, info, left_foot, right_foot):
    d=state.data; torso_up=d.xmat[env.torso_body,:,2]
    board_level_progress=jnp.clip(1.-jnp.abs(info["board_tilt_rad"])/BOARD_INITIAL_TILT,0.,1.)
    board_q=(1.-BOARD_TILT_CURVATURE)*board_level_progress+BOARD_TILT_CURVATURE*jnp.square(board_level_progress)
    # Far from level, allow the board to move so the policy can correct its
    # initial tilt.  Inside roughly +/-5 degrees, smoothly strengthen the
    # speed penalty so that crossing level at high speed is costly.  A smooth
    # square-root cost is deliberately sensitive to small residual speed but
    # compresses already-large speed.  Subtracting sqrt(epsilon) keeps the
    # penalty exactly zero at zero angular speed and avoids a singular slope.
    board_speed_near_level=jax.nn.sigmoid(
        (BOARD_ANGULAR_SPEED_NEAR_LEVEL_ANGLE-jnp.abs(info["board_tilt_rad"]))
        / BOARD_ANGULAR_SPEED_TRANSITION_WIDTH
    )
    board_speed_weight=BOARD_ANGULAR_SPEED_FAR_WEIGHT+(
        BOARD_ANGULAR_SPEED_NEAR_WEIGHT-BOARD_ANGULAR_SPEED_FAR_WEIGHT
    )*board_speed_near_level
    normalized_board_speed=info["board_angular_speed"]/BOARD_ANGULAR_SPEED_SCALE
    board_speed_cost=(
        jnp.sqrt(normalized_board_speed+BOARD_ANGULAR_SPEED_SQRT_EPSILON)
        - jnp.sqrt(BOARD_ANGULAR_SPEED_SQRT_EPSILON)
    )
    torso_z_cos=jnp.clip(torso_up[2],-1.,1.);torso_z_q=_quality(jnp.sqrt(jnp.maximum(0.,1.-torso_z_cos**2)),TORSO_Z_SIN_SCALE);torso_y_q=_quality(jnp.abs(torso_up[1]),TORSO_Y_COS_SCALE)
    contact_q=.5*(info["left_foot_on_board"]+info["right_foot_on_board"])
    slip=.5*(jnp.linalg.norm((left_foot-state.previous_left_foot)[:2]/env.control_dt)+jnp.linalg.norm((right_foot-state.previous_right_foot)[:2]/env.control_dt));slip_q=_quality(slip,FOOT_SLIP_SCALE)
    lq=_quality(info["left_foot_target_xy_error"],FOOT_TARGET_XY_SCALE)*_quality(info["left_foot_target_z_error"],FOOT_TARGET_Z_SCALE);rq=_quality(info["right_foot_target_xy_error"],FOOT_TARGET_XY_SCALE)*_quality(info["right_foot_target_z_error"],FOOT_TARGET_Z_SCALE);foot_q=.5*(lq+rq)
    masses=env.robot_masses;com=jnp.sum(d.xipos[env.robot_body_ids]*masses[:,None],axis=0)/jnp.sum(masses);pivot=d.xpos[env.roller_body]+env.roller_radius*d.xmat[env.board_body,:,2];com_error=jnp.linalg.norm(com[:2]-pivot[:2]);com_q=_quality(com_error,COM_PIVOT_HORIZONTAL_SCALE)
    relative_height=info["pelvis_z"]-d.xpos[env.board_body,2]
    knee_indices=jnp.asarray([env.left_knee_index,env.right_knee_index])
    knee_angles=d.qpos[env.qpos_ids[knee_indices]]
    knee_reference=env.default_joint_pos[knee_indices]
    knee_angle_change=jnp.abs(knee_angles-knee_reference)
    knee_excess=jnp.maximum(knee_angle_change-KNEE_CHANGE_FREE_ANGLE,0.)
    knee_change_cost=jnp.mean(jnp.square(knee_excess/KNEE_CHANGE_SCALE))
    action_rate=jnp.sqrt(jnp.mean((action-state.last_action)**2));action_q=_quality(action_rate,ACTION_RATE_SCALE);joint_rms=jnp.sqrt(jnp.mean(d.qvel[env.qvel_ids]**2));joint_q=_quality(joint_rms,JOINT_VELOCITY_SCALE)
    torso_index=env.torso_action_index
    torso_velocity=jnp.abs(d.qvel[env.qvel_ids[torso_index]])
    torso_action_rate=jnp.abs(action[torso_index]-state.last_action[torso_index])
    torso_action_magnitude=jnp.abs(action[torso_index])
    torso_velocity_q=_quality(torso_velocity,TORSO_JOINT_VELOCITY_SCALE)
    torso_action_rate_q=_quality(torso_action_rate,TORSO_ACTION_RATE_SCALE)
    torso_action_magnitude_q=_quality(torso_action_magnitude,TORSO_ACTION_MAGNITUDE_SCALE)
    terms={"alive_reward":ALIVE_REWARD_PER_STEP*(~info["physical_failure"]),"board_tilt_reward":BOARD_TILT_WEIGHT*board_q,"board_angular_speed_penalty":-board_speed_weight*board_speed_cost,"torso_z_reward":torso_z_q,"torso_y_reward":torso_y_q,"foot_slip_reward":.5*slip_q,"foot_target_reward":.5*foot_q,"com_pivot_reward":.5*com_q,"knee_change_penalty":-KNEE_CHANGE_WEIGHT*knee_change_cost,"action_rate_reward":.3*action_q,"joint_velocity_reward":.3*joint_q,"torso_joint_velocity_penalty":-TORSO_JOINT_VELOCITY_WEIGHT*(1.-torso_velocity_q),"torso_action_rate_penalty":-TORSO_ACTION_RATE_WEIGHT*(1.-torso_action_rate_q),"torso_action_magnitude_penalty":-TORSO_ACTION_MAGNITUDE_WEIGHT*(1.-torso_action_magnitude_q),"success_reward":10.*info["is_success"],"failure_penalty":-10.*info["physical_failure"],"timeout_penalty":-2.*info["timeout"]}
    reward=jnp.clip(jnp.nan_to_num(sum(terms.values()),nan=-10.),-10.,10.)
    diagnostics={**terms,"board_level_progress":board_level_progress,"board_tilt_quality":board_q,"board_angular_speed_penalty_weight":board_speed_weight,"board_angular_speed_near_level_gate":board_speed_near_level,"contact_quality":contact_q,"foot_slip_speed":slip,"foot_target_quality":foot_q,"com_pivot_horizontal_error":com_error,"pelvis_board_relative_height":relative_height,"knee_change_cost":knee_change_cost,"left_knee_angle":knee_angles[0],"right_knee_angle":knee_angles[1],"left_knee_angle_change":knee_angle_change[0],"right_knee_angle_change":knee_angle_change[1],"joint_speed":joint_rms,"torso_joint_velocity":torso_velocity,"torso_action_rate":torso_action_rate,"torso_action_magnitude":torso_action_magnitude,"total_reward":reward}
    return reward,diagnostics
