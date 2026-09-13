"""Batched JAX/MJX implementation of the Mount1 task."""

from __future__ import annotations

from pathlib import Path

from flax import struct
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from rewards.mount1_rewards import mount1_reward


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_JOINTS = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
]
ACTION_SCALES = np.array(
    [0.35, 0.60, 0.40, 0.70, 0.30, 0.25] * 2 + [0.10], dtype=np.float64
)
INIT_BASE_POS = np.array([0.01896389, -0.02813489, 1.08841177])
INIT_BASE_QUAT = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
INIT_JOINT_POS = {
    "left_hip_yaw_joint": -0.03349077, "left_hip_pitch_joint": -0.47352900,
    "left_hip_roll_joint": -0.00019824, "left_knee_joint": 0.98734730,
    "left_ankle_pitch_joint": -0.50477822, "left_ankle_roll_joint": 0.26179900,
    "right_hip_yaw_joint": -0.02353854, "right_hip_pitch_joint": -0.15366925,
    "right_hip_roll_joint": 0.01337735, "right_knee_joint": 0.33320697,
    "right_ankle_pitch_joint": -0.17375228, "right_ankle_roll_joint": 0.24850160,
    "left_shoulder_roll_joint": 0.15, "right_shoulder_roll_joint": -0.15,
    "left_elbow_joint": np.deg2rad(80.0), "right_elbow_joint": np.deg2rad(80.0),
}
LEFT_FOOT_TARGET_IN_BOARD = np.array([-0.163, 0.0, 0.0158])
RIGHT_FOOT_TARGET_IN_BOARD = np.array([0.163, 0.0, 0.0158])
OBSERVATION_DIM = 93

SUCCESS_FOOT_SAFE_X, SUCCESS_FOOT_SAFE_Y = 0.30, 0.04
SUCCESS_FOOT_Z_MIN, SUCCESS_FOOT_Z_MAX = 0.0, 0.05
SUCCESS_TORSO_TILT_MAX = np.deg2rad(3.0)
SUCCESS_PELVIS_VERTICAL_SPEED_MAX = 0.30
SUCCESS_JOINT_SPEED_MAX = 0.50
SUCCESS_TORSO_ANGULAR_SPEED_MAX = 0.30
SUCCESS_BOARD_TILT_MAX = np.deg2rad(3.0)
SUCCESS_BOARD_ANGULAR_SPEED_MAX = 0.30
SUCCESS_BOARD_ROLLER_HORIZONTAL_ERROR_MAX = 0.15
SUCCESS_ROLLER_AXIS_ERROR_MAX = np.deg2rad(2.0)
SUCCESS_HOLD_SECONDS = 0.50
SUCCESS_LEFT_FOOT_TARGET_XY_ERROR_MAX = 0.05
SUCCESS_LEFT_FOOT_TARGET_Z_ERROR_MAX = 0.03
SUCCESS_RIGHT_FOOT_TARGET_XY_ERROR_MAX = 0.05
SUCCESS_RIGHT_FOOT_TARGET_Z_ERROR_MAX = 0.03

FAILURE_PELVIS_Z_MIN = 0.55
FAILURE_TORSO_TILT_MAX = np.deg2rad(70.0)
FAILURE_BOARD_TILT_MAX = np.deg2rad(70.0)
FAILURE_BOARD_ROLLER_HORIZONTAL_ERROR_MAX = 0.35
FAILURE_ROLLER_AXIS_ERROR_MAX = np.deg2rad(60.0)
FAILURE_BOARD_ROLLER_LOST_CONTACT_SECONDS = 0.20
FAILURE_ROBOT_BOARD_HORIZONTAL_DISTANCE_MAX = 1.20
FAILURE_ABS_QVEL_MAX = 100.0


@struct.dataclass
class Mount1State:
    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]
    step_count: jax.Array
    success_hold_steps: jax.Array
    board_lost_steps: jax.Array
    last_action: jax.Array
    previous_left_foot: jax.Array
    previous_right_foot: jax.Array
    kp: jax.Array
    kd: jax.Array
    rng: jax.Array
    disturbance_force: jax.Array
    disturbance_steps_remaining: jax.Array
    next_disturbance_step: jax.Array


class Mount1Env:
    """Static MJX model plus pure single-environment reset/step functions."""

    def __init__(self, config) -> None:
        self.control_dt = 0.02
        self.episode_seconds = float(config.episode_seconds)
        self.frame_skip = 20
        self.max_steps = int(round(self.episode_seconds / self.control_dt))
        self.success_hold_required = int(round(SUCCESS_HOLD_SECONDS / self.control_dt))
        self.board_lost_max = int(round(FAILURE_BOARD_ROLLER_LOST_CONTACT_SECONDS / self.control_dt))
        self.reset_position_noise = float(config.reset_position_noise)
        self.reset_velocity_noise = float(config.reset_velocity_noise)
        self.reset_base_xy_noise = float(config.reset_base_xy_noise)
        self.pd_gain_randomization = float(config.pd_gain_randomization)
        self.disturbance_force_max = float(config.disturbance_force_max)
        self.disturbance_interval_steps = max(1, int(round(float(config.disturbance_interval_seconds) / self.control_dt)))
        self.disturbance_duration_steps = max(1, int(round(float(config.disturbance_duration_seconds) / self.control_dt)))

        model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models" / "mount1.xml"))
        # MJX 0.3.x does not implement cylinder-box or cylinder-mesh pairs.
        # Only the feet are legal robot contacts in this task, so disable the
        # other robot collision proxies and use an equivalent roller capsule.
        pelvis = model.body("pelvis").id
        robot_bodies = set()
        for body in range(model.nbody):
            parent = body
            while parent:
                if parent == pelvis:
                    robot_bodies.add(body)
                    break
                parent = int(model.body_parentid[parent])
        foot_geoms = {model.geom("left_foot_collision").id, model.geom("right_foot_collision").id}
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) in robot_bodies and geom not in foot_geoms:
                model.geom_contype[geom] = 0
                model.geom_conaffinity[geom] = 0
        roller_geom = model.geom("roller_geom").id
        model.geom_type[roller_geom] = mujoco.mjtGeom.mjGEOM_CAPSULE
        model.geom_size[roller_geom, 1] = 0.10  # 0.10 cylinder half-length + two 0.05 caps

        self.cpu_model = model
        self.model = mjx.put_model(model)
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
        self.pelvis_body, self.torso_body = pelvis, model.body("torso_link").id
        self.board_body, self.roller_body = model.body("board").id, model.body("roller").id
        self.left_site, self.right_site = model.site("left_foot").id, model.site("right_foot").id
        self.floor_geom, self.board_geom, self.roller_geom = model.geom("floor").id, model.geom("board_geom").id, roller_geom
        self.left_geom, self.right_geom = tuple(foot_geoms)
        if self.left_geom != model.geom("left_foot_collision").id:
            self.left_geom, self.right_geom = self.right_geom, self.left_geom

        joint_qpos = lambda n: int(model.jnt_qposadr[model.joint(n).id])
        joint_dof = lambda n: int(model.jnt_dofadr[model.joint(n).id])
        self.base_qpos, self.base_dof = joint_qpos("floating_base_joint"), joint_dof("floating_base_joint")
        self.board_dof, self.roller_dof = joint_dof("board_free"), joint_dof("roller_free")
        self.qpos_ids = jnp.asarray([joint_qpos(n) for n in CONTROLLED_JOINTS])
        self.qvel_ids = jnp.asarray([joint_dof(n) for n in CONTROLLED_JOINTS])
        actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        self.all_qpos_ids = jnp.asarray([joint_qpos(n) for n in actuator_names])
        self.all_qvel_ids = jnp.asarray([joint_dof(n) for n in actuator_names])
        self.actuator_ids = jnp.asarray([model.actuator(n).id for n in CONTROLLED_JOINTS])
        self.ctrl_low = jnp.asarray(model.actuator_ctrlrange[:, 0])
        self.ctrl_high = jnp.asarray(model.actuator_ctrlrange[:, 1])
        self.action_scales = jnp.asarray(ACTION_SCALES)
        self.left_knee_index = CONTROLLED_JOINTS.index("left_knee_joint")
        self.right_knee_index = CONTROLLED_JOINTS.index("right_knee_joint")
        self.torso_action_index = CONTROLLED_JOINTS.index("torso_joint")
        self.left_target = jnp.asarray(LEFT_FOOT_TARGET_IN_BOARD)
        self.right_target = jnp.asarray(RIGHT_FOOT_TARGET_IN_BOARD)

        init_qpos = model.qpos0.copy()
        init_qpos[self.base_qpos:self.base_qpos + 3] = INIT_BASE_POS
        init_qpos[self.base_qpos + 3:self.base_qpos + 7] = INIT_BASE_QUAT / np.linalg.norm(INIT_BASE_QUAT)
        for name, value in INIT_JOINT_POS.items():
            init_qpos[joint_qpos(name)] = value
        self.init_qpos = jnp.asarray(init_qpos)
        self.default_joint_pos = self.init_qpos[self.qpos_ids]
        self.all_default_joint_pos = self.init_qpos[self.all_qpos_ids]

        kp, kd = np.full(model.nu, 45.0), np.full(model.nu, 2.0)
        for i, name in enumerate(actuator_names):
            if "ankle" in name: kp[i], kd[i] = 30.0, 1.2
            elif any(x in name for x in ("shoulder", "elbow", "wrist")): kp[i], kd[i] = 40.0, 4.0
            elif name.startswith(("L_", "R_")): kp[i], kd[i] = 1.0, 0.05
        self.nominal_kp, self.nominal_kd = jnp.asarray(kp), jnp.asarray(kd)
        masses = model.body_mass[np.asarray(sorted(robot_bodies))]
        self.robot_body_ids = jnp.asarray(sorted(robot_bodies))
        self.robot_masses = jnp.asarray(masses)
        self.roller_radius = float(model.geom_size[roller_geom, 0])

        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray(self.init_qpos)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        self.data_template = mjx.put_data(model, data)

    @staticmethod
    def _tilt(rotation):
        return jnp.arccos(jnp.clip(rotation[2, 2], -1.0, 1.0))

    @staticmethod
    def _point_in_frame(point, origin, rotation):
        return rotation.T @ (point - origin)

    def _touching(self, data, geom_a, geom_b):
        pairs = data.contact.geom
        pair = ((pairs[:, 0] == geom_a) & (pairs[:, 1] == geom_b)) | ((pairs[:, 1] == geom_a) & (pairs[:, 0] == geom_b))
        return jnp.any(pair & (data.contact.dist <= 0.0))

    def reset(self, key):
        key_pos, key_vel, key_xy, key_gain = jax.random.split(key, 4)
        qpos = self.init_qpos.at[self.qpos_ids].add(jax.random.uniform(key_pos, (13,), minval=-self.reset_position_noise, maxval=self.reset_position_noise))
        qpos = qpos.at[self.base_qpos:self.base_qpos + 2].add(jax.random.uniform(key_xy, (2,), minval=-self.reset_base_xy_noise, maxval=self.reset_base_xy_noise))
        qvel = jnp.zeros((self.nv,)).at[self.qvel_ids].set(jax.random.uniform(key_vel, (13,), minval=-self.reset_velocity_noise, maxval=self.reset_velocity_noise))
        gain = jax.random.uniform(key_gain, (self.nu,), minval=1.0-self.pd_gain_randomization, maxval=1.0+self.pd_gain_randomization)
        data = self.data_template.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros((self.nu,)), xfrc_applied=jnp.zeros((self.cpu_model.nbody, 6)))
        data = mjx.forward(self.model, data)
        left, right = data.site_xpos[self.left_site], data.site_xpos[self.right_site]
        next_push = jnp.where(
            self.disturbance_force_max > 0,
            jax.random.randint(key_gain, (), max(1, self.disturbance_interval_steps // 2), self.disturbance_interval_steps + 1),
            self.max_steps + 1,
        )
        state = Mount1State(data, jnp.zeros((OBSERVATION_DIM,)), jnp.array(0.0), jnp.array(False), {}, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), jnp.array(0, jnp.int32), jnp.zeros((13,)), left, right, self.nominal_kp*gain, self.nominal_kd*jnp.sqrt(gain), key, jnp.zeros((3,)), jnp.array(0, jnp.int32), next_push)
        metrics = self._metrics(state, left, right)
        metrics = {**metrics, "is_success": jnp.array(False),
                   "physical_failure": jnp.array(False),
                   "timeout": jnp.array(False), "episode_seconds": jnp.array(0.0)}
        reward, reward_metrics = mount1_reward(
            self, state, jnp.zeros((13,)), metrics, left, right
        )
        metrics = {**metrics, **reward_metrics}
        return state.replace(obs=self._obs(state, metrics, left, right),
                             reward=reward, metrics=metrics)

    def _pd_step(self, data, action, kp, kd):
        targets = self.all_default_joint_pos.at[self.actuator_ids].set(self.default_joint_pos + self.action_scales * action)
        torque = kp * (targets - data.qpos[self.all_qpos_ids]) - kd * data.qvel[self.all_qvel_ids] + data.qfrc_bias[self.all_qvel_ids]
        return mjx.step(self.model, data.replace(ctrl=jnp.clip(torque, self.ctrl_low, self.ctrl_high)))

    def step(self, state, action):
        action = jnp.clip(action, -1.0, 1.0)
        rng, push_key, interval_key = jax.random.split(state.rng, 3)
        active = state.disturbance_steps_remaining > 0
        trigger = (~active) & (state.step_count >= state.next_disturbance_step) & (self.disturbance_force_max > 0)
        angle = jax.random.uniform(push_key, (), minval=0.0, maxval=2*jnp.pi)
        magnitude = jax.random.uniform(push_key, (), minval=.25, maxval=1.0) * self.disturbance_force_max
        sampled_force = magnitude*jnp.array([jnp.cos(angle), jnp.sin(angle), 0.0])
        force = jnp.where(trigger, sampled_force, jnp.where(active, state.disturbance_force, jnp.zeros((3,))))
        remaining = jnp.where(trigger, self.disturbance_duration_steps-1, jnp.maximum(state.disturbance_steps_remaining-1, 0))
        next_push = jnp.where(trigger, state.step_count+jax.random.randint(interval_key, (), max(1,self.disturbance_interval_steps//2), self.disturbance_interval_steps+1), state.next_disturbance_step)
        xfrc = jnp.zeros_like(state.data.xfrc_applied).at[self.pelvis_body,:3].set(force)
        data0 = state.data.replace(xfrc_applied=xfrc)
        data = jax.lax.fori_loop(0, self.frame_skip, lambda _, d: self._pd_step(d, action, state.kp, state.kd), data0)
        left, right = data.site_xpos[self.left_site], data.site_xpos[self.right_site]
        provisional = state.replace(data=data, step_count=state.step_count+1)
        metrics = self._metrics(provisional, left, right)
        success_hold = jnp.where(metrics["success_conditions_now"], state.success_hold_steps+1, 0)
        board_lost = jnp.where(metrics["board_roller_contact"], 0, state.board_lost_steps+1)
        success = success_hold >= self.success_hold_required
        physical_failure = metrics["robot_fallen"] | metrics["board_failed"] | metrics["out_of_area"] | metrics["numerical_failure"] | (board_lost >= self.board_lost_max)
        timeout = (provisional.step_count >= self.max_steps) & ~success & ~physical_failure
        done = success | physical_failure | timeout
        metrics = {**metrics, "is_success": success, "physical_failure": physical_failure, "timeout": timeout, "episode_seconds": provisional.step_count*self.control_dt}
        reward, reward_metrics = mount1_reward(self, provisional, action, metrics, left, right)
        metrics = {**metrics, **reward_metrics}
        updated = provisional.replace(reward=reward, done=done, metrics=metrics, success_hold_steps=success_hold, board_lost_steps=board_lost, last_action=action, previous_left_foot=left, previous_right_foot=right, rng=rng, disturbance_force=force, disturbance_steps_remaining=remaining, next_disturbance_step=next_push)
        return updated.replace(obs=self._obs(updated, metrics, left, right))

    def _metrics(self, state, left, right):
        d=state.data; br=d.xmat[self.board_body]; bp=d.xpos[self.board_body]
        lf=self._point_in_frame(left,bp,br); rf=self._point_in_frame(right,bp,br); roller=self._point_in_frame(d.xpos[self.roller_body],bp,br)
        left_board=self._touching(d,self.left_geom,self.board_geom); right_board=self._touching(d,self.right_geom,self.board_geom)
        board_roller=self._touching(d,self.board_geom,self.roller_geom)
        torso_tilt=self._tilt(d.xmat[self.torso_body]);board_tilt=self._tilt(br)
        board_w=jnp.linalg.norm(d.cvel[self.board_body,:3]);torso_w=jnp.linalg.norm(d.cvel[self.torso_body,:3]);joint_speed=jnp.max(jnp.abs(d.qvel[self.qvel_ids]))
        axis=jnp.abs(jnp.dot(br[:,1],d.xmat[self.roller_body][:,2]));axis_error=jnp.arccos(jnp.clip(axis,0,1));roller_error=jnp.linalg.norm(roller[:2])
        lfxy=jnp.linalg.norm(lf[:2]-self.left_target[:2]);rfxy=jnp.linalg.norm(rf[:2]-self.right_target[:2]);lfz=jnp.abs(lf[2]-self.left_target[2]);rfz=jnp.abs(rf[2]-self.right_target[2])
        feet_ok=left_board & right_board & (jnp.abs(lf[0])<SUCCESS_FOOT_SAFE_X)&(jnp.abs(rf[0])<SUCCESS_FOOT_SAFE_X)&(jnp.abs(lf[1])<SUCCESS_FOOT_SAFE_Y)&(jnp.abs(rf[1])<SUCCESS_FOOT_SAFE_Y)&(lf[2]>SUCCESS_FOOT_Z_MIN)&(rf[2]>SUCCESS_FOOT_Z_MIN)&(lf[2]<SUCCESS_FOOT_Z_MAX)&(rf[2]<SUCCESS_FOOT_Z_MAX)
        success_now=feet_ok&(lfxy<SUCCESS_LEFT_FOOT_TARGET_XY_ERROR_MAX)&(rfxy<SUCCESS_RIGHT_FOOT_TARGET_XY_ERROR_MAX)&(lfz<SUCCESS_LEFT_FOOT_TARGET_Z_ERROR_MAX)&(rfz<SUCCESS_RIGHT_FOOT_TARGET_Z_ERROR_MAX)&(torso_tilt<SUCCESS_TORSO_TILT_MAX)&(jnp.abs(d.qvel[self.base_dof+2])<SUCCESS_PELVIS_VERTICAL_SPEED_MAX)&(joint_speed<SUCCESS_JOINT_SPEED_MAX)&(torso_w<SUCCESS_TORSO_ANGULAR_SPEED_MAX)&(board_tilt<SUCCESS_BOARD_TILT_MAX)&(board_w<SUCCESS_BOARD_ANGULAR_SPEED_MAX)&(roller_error<SUCCESS_BOARD_ROLLER_HORIZONTAL_ERROR_MAX)&(axis_error<SUCCESS_ROLLER_AXIS_ERROR_MAX)
        fallen=(d.xpos[self.pelvis_body,2]<FAILURE_PELVIS_Z_MIN)|(torso_tilt>FAILURE_TORSO_TILT_MAX)
        board_failed=(board_tilt>FAILURE_BOARD_TILT_MAX)|(roller_error>FAILURE_BOARD_ROLLER_HORIZONTAL_ERROR_MAX)|(axis_error>FAILURE_ROLLER_AXIS_ERROR_MAX)
        out=jnp.linalg.norm(d.xpos[self.pelvis_body,:2]-bp[:2])>FAILURE_ROBOT_BOARD_HORIZONTAL_DISTANCE_MAX
        finite=jnp.all(jnp.isfinite(d.qpos))&jnp.all(jnp.isfinite(d.qvel)); numerical=(~finite)|(jnp.max(jnp.abs(d.qvel))>FAILURE_ABS_QVEL_MAX)
        return {"left_foot_on_board":left_board,"right_foot_on_board":right_board,"board_roller_contact":board_roller,"left_foot_target_xy_error":lfxy,"right_foot_target_xy_error":rfxy,"left_foot_target_z_error":lfz,"right_foot_target_z_error":rfz,"pelvis_z":d.xpos[self.pelvis_body,2],"pelvis_vertical_speed":d.qvel[self.base_dof+2],"controlled_joint_speed":joint_speed,"torso_tilt_rad":torso_tilt,"torso_angular_speed":torso_w,"board_tilt_rad":board_tilt,"board_angular_speed":board_w,"board_roller_horizontal_error":roller_error,"roller_axis_error_rad":axis_error,"success_conditions_now":success_now,"robot_fallen":fallen,"board_failed":board_failed,"out_of_area":out,"numerical_failure":numerical}

    def _obs(self,state,m,left,right):
        d=state.data;pr=d.xmat[self.pelvis_body];br=d.xmat[self.board_body];forward=pr[:,0]*jnp.array([1.,1.,0.]);forward=forward/jnp.maximum(jnp.linalg.norm(forward),1e-8);heading=jnp.stack([forward,jnp.array([-forward[1],forward[0],0.]),jnp.array([0.,0.,1.])],axis=1)
        pa,pv=d.cvel[self.pelvis_body,:3],d.cvel[self.pelvis_body,3:];ba,bv=d.cvel[self.board_body,:3],d.cvel[self.board_body,3:];ra,rv=d.cvel[self.roller_body,:3],d.cvel[self.roller_body,3:]
        bp=d.xpos[self.board_body];rp=d.xpos[self.roller_body];lf=self._point_in_frame(left,bp,br);rf=self._point_in_frame(right,bp,br)
        lv=br.T@((left-state.previous_left_foot)/self.control_dt-bv);rvf=br.T@((right-state.previous_right_foot)/self.control_dt-bv)
        obs=jnp.concatenate([d.qpos[self.qpos_ids]-self.default_joint_pos,d.qvel[self.qvel_ids],pr.T@jnp.array([0.,0.,-1.]),pr.T@pv,pr.T@pa,jnp.array([d.xpos[self.pelvis_body,2]]),state.last_action,heading.T@(bp-d.xpos[self.pelvis_body]),heading.T@br[:,0],heading.T@br[:,2],heading.T@(bv-pv),heading.T@(ba-pa),self._point_in_frame(rp,bp,br),br.T@d.xmat[self.roller_body][:,2],br.T@(rv-bv),br.T@(ra-ba),lf-self.left_target,rf-self.right_target,lv,rvf,jnp.array([self._touching(d,self.left_geom,self.floor_geom),self._touching(d,self.right_geom,self.floor_geom),m["left_foot_on_board"],m["right_foot_on_board"]]),jnp.array([jnp.minimum(state.step_count/self.max_steps,1.)])])
        return jnp.nan_to_num(obs,nan=0.,posinf=10.,neginf=-10.)
