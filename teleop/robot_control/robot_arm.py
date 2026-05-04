import os
import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import ( LowCmd_  as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import ( LowCmd_  as go_LowCmd, LowState_ as go_LowState)  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

def _try_open_hands(duration: float = 1.0):
    """Best-effort: make Dex3-1 fingers compact before arm motion."""
    try:
        from teleop.robot_control.robot_hand_unitree import dex3_prepare_safe_hands
        dex3_prepare_safe_hands(open_duration=duration, release_duration=0.4)
    except Exception as e:
        logger_mp.debug(f"[_try_open_hands] skipped: {e}")

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

# ─── Gravity Compensation (Pinocchio) ─────────────────────────────────────
_GRAV_URDF_PATH = (
    "/home/humanoid-pc/unitree_rl_gym/resources/robots/"
    "g1_description/g1_29dof_with_hand_rev_1_0.urdf"
)
_UNITREE_TO_PIN = {}
for _i in range(15):
    _UNITREE_TO_PIN[_i] = _i
for _i in range(7):
    _UNITREE_TO_PIN[15 + _i] = 15 + _i
    _UNITREE_TO_PIN[22 + _i] = 29 + _i

_WAIST_JOINTS = [12, 13, 14]
_ARM_JOINTS = list(range(15, 29))


class GravityCompensator:
    """Compute per-joint gravity torques using Pinocchio (full model)."""

    def __init__(self):
        self.available = False
        try:
            import pinocchio as pin
            self.pin = pin
            self.model = pin.buildModelFromUrdf(_GRAV_URDF_PATH)
            self.data = self.model.createData()
            self.neutral_q = pin.neutral(self.model)
            self.available = True
            logger_mp.info("Gravity compensation: ENABLED (Pinocchio + URDF)")
        except Exception as e:
            logger_mp.warning(f"Gravity compensation: DISABLED ({e})")

    def compute(self, motor_q_func):
        """Return {unitree_joint_idx: tau_ff} for waist + arm joints."""
        if not self.available:
            return {}
        q = self.neutral_q.copy()
        for u_idx, p_idx in _UNITREE_TO_PIN.items():
            if p_idx < self.model.nq:
                q[p_idx] = motor_q_func(u_idx)
        G = self.pin.computeGeneralizedGravity(self.model, self.data, q)
        tau_ff = {}
        for j in _WAIST_JOINTS + _ARM_JOINTS:
            p_idx = _UNITREE_TO_PIN[j]
            tau_ff[j] = float(G[p_idx])
        return tau_ff

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20
 

_MOTOR_TEMP_WARN = 70
_MOTOR_TEMP_DISABLE = 85

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None
        self.temperature = (0, 0)
        self.mode = 1

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_Num_Motors)]

class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

_G1_29_ARM_Q_LOWER = np.array([
    # left arm — shoulder/elbow at 85% URDF, wrist at 60% URDF
    -2.626, -1.350, -2.225, -0.890, -1.183, -0.968, -0.968,
    # right arm
    -2.626, -1.914, -2.225, -0.890, -1.183, -0.968, -0.968,
])
_G1_29_ARM_Q_UPPER = np.array([
    # left arm
     2.270,  1.914,  2.225,  1.780,  1.183,  0.968,  0.968,
    # right arm
     2.270,  1.350,  2.225,  1.780,  1.183,  0.968,  0.968,
])

# Disabled arm joint indices (14-dim arm space). These joints are locked to 0.
# Index map: 0=L_ShoulderPitch 1=L_ShoulderRoll 2=L_ShoulderYaw 3=L_Elbow
#            4=L_WristRoll     5=L_WristPitch   6=L_WristYaw
#            7=R_ShoulderPitch 8=R_ShoulderRoll 9=R_ShoulderYaw 10=R_Elbow
#           11=R_WristRoll    12=R_WristPitch  13=R_WristYaw
# 2026-04-27: motor 20 (L_WristPitch) replaced with new hardware; previously
# disabled set {5} reverted to empty. Keep the variable so we can re-disable a
# joint quickly if another fault appears. See todo_docs/motor20_wrist_pitch_fault_report.md.
_G1_29_DISABLED_ARM_JOINTS: set[int] = set()

_SPREAD_Q = np.zeros(14)
_SPREAD_Q[1] = 1.5    # L_ShoulderRoll → outward
_SPREAD_Q[8] = -1.5   # R_ShoulderRoll → outward

_CLEARANCE_Q = np.zeros(14)
_CLEARANCE_Q[1] = 1.75   # extra outward clearance for default-pose parking
_CLEARANCE_Q[8] = -1.75

# /tmp PID-file used by utils/arm_idle_holder.py to know when to yield.
# Teleop writes its own PID here on startup so the holder stops fighting
# us; we remove the file on clean shutdown so the holder resumes.
_HOLDER_YIELD_FLAG_PATH = "/tmp/g1_arm_holder_yield.pid"


def _publish_yield_flag():
    """Write our PID into the yield flag so arm_idle_holder pauses."""
    try:
        with open(_HOLDER_YIELD_FLAG_PATH, "w") as f:
            f.write(f"{os.getpid()}\n")
    except OSError as e:
        logger_mp.warning(f"[holder-flag] could not write {_HOLDER_YIELD_FLAG_PATH}: {e}")


def _clear_yield_flag():
    """Remove the yield flag so arm_idle_holder resumes spread-pose hold."""
    try:
        os.remove(_HOLDER_YIELD_FLAG_PATH)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger_mp.warning(f"[holder-flag] could not remove {_HOLDER_YIELD_FLAG_PATH}: {e}")


def _keep_holder_yielding_until_next_teleop():
    """Pause arm_idle_holder after exit by writing pid 1 as the yield owner."""
    try:
        with open(_HOLDER_YIELD_FLAG_PATH, "w") as f:
            f.write("1\n")
        logger_mp.warning(
            "[holder-flag] arm_idle_holder will stay yielded until the next "
            "teleop/RL process overwrites the flag or the flag is removed."
        )
    except OSError as e:
        logger_mp.warning(f"[holder-flag] could not pause holder: {e}")


class G1_29_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False,
                 safe_deploy = True, keep_spread = True,
                 safe_deploy_q = None,
                 safe_deploy_via_q = None,
                 safe_deploy_via_min_duration = 1.0,
                 safe_deploy_min_duration = 1.0,
                 safe_deploy_after_via_callback = None,
                 prepare_hands_on_deploy = True,
                 wrist_kp = 60.0,
                 wrist_kd = 2.0):
        """Args:
            motion_mode: publish to rt/arm_sdk (True) or rt/lowcmd (False).
            simulation_mode: skip clip_arm_q_target & velocity limit.
            safe_deploy: in __init__, open hands then spread shoulders
                outward before user code runs. Strongly recommended for
                G1+Dex3 to clear the hands from the body.
            safe_deploy_q: optional 14D target for the safe-deploy hold pose.
                The default is _SPREAD_Q. Single-arm teleop can pass a mixed
                pose so the inactive arm does not first move to full spread.
            safe_deploy_via_q: optional 14D waypoint to visit before
                safe_deploy_q. Single-arm teleop uses an outward active-arm
                waypoint to avoid sweeping directly from relaxed/down to q=0.
            safe_deploy_via_min_duration/safe_deploy_min_duration: timing for
                the optional waypoint and final deploy target.
            safe_deploy_after_via_callback: optional callback after the
                outward waypoint is reached. Used to open Dex3 hands only
                after the fingers have cleared the legs.
            prepare_hands_on_deploy: when False, skip the internal Dex3 hand
                close/release step because an external hand controller is
                already holding the fingers.
            wrist_kp/wrist_kd: lower wrist PD gains for smoother controller
                rotation tracking near the hand.
            keep_spread: when True (default 2026-04-27), the safe-deploy
                phase 2 ("move to home q=0") is SKIPPED. Reason: the
                default standing pose collides Dex3-1 fingers with the
                outer thighs; staying at spread keeps them safe. The
                companion daemon utils/arm_idle_holder.py will continue
                to hold this pose between teleop sessions. See
                todo_docs/dex3_hand_error.md.
        """
        logger_mp.info("Initialize G1_29_ArmController...")
        self.keep_spread = keep_spread
        self.safe_deploy_q = (
            np.asarray(safe_deploy_q, dtype=np.float64).copy()
            if safe_deploy_q is not None
            else _SPREAD_Q.copy()
        )
        self.safe_deploy_via_q = (
            np.asarray(safe_deploy_via_q, dtype=np.float64).copy()
            if safe_deploy_via_q is not None
            else None
        )
        self.safe_deploy_via_min_duration = float(safe_deploy_via_min_duration)
        self.safe_deploy_min_duration = float(safe_deploy_min_duration)
        self.safe_deploy_after_via_callback = safe_deploy_after_via_callback
        self.prepare_hands_on_deploy = bool(prepare_hands_on_deploy)
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 150.0
        self.kd_low = 3.5
        self.kp_wrist = float(wrist_kp)
        self.kd_wrist = float(wrist_kd)
        self.kp_waist = 200.0
        self.kd_waist = 5.0

        self.grav_comp = GravityCompensator()

        self.all_motor_q = None
        self.arm_velocity_limit = 25.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        self.q_target = self.get_current_dual_arm_q().copy()
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        waist_indices = set(_WAIST_JOINTS)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            elif id.value in waist_indices:
                self.msg.motor_cmd[id].kp = self.kp_waist
                self.msg.motor_cmd[id].kd = self.kd_waist
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]

        for j in _WAIST_JOINTS:
            self.msg.motor_cmd[j].q = 0.0
        logger_mp.info(f"Lock OK! (waist kp={self.kp_waist}, kd={self.kd_waist}, q=0.0)")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        # Start our publisher before asking arm_idle_holder to yield. This
        # avoids a brief no-publisher window where the arms can sag under
        # gravity at teleop startup.
        if self.motion_mode:
            pre_yield_q = self.get_current_dual_arm_q().copy()
            with self.ctrl_lock:
                self.q_target = pre_yield_q.copy()
            logger_mp.info("[holder-flag] teleop publisher warmup before yield...")
            time.sleep(0.75)
            _publish_yield_flag()
            logger_mp.info("[holder-flag] arm_idle_holder yielded to teleop.")
            # Keep commanding the pre-yield pose long enough for arm_sdk
            # ownership to settle before any deploy waypoint starts moving.
            hold_until = time.time() + 1.5
            while time.time() < hold_until:
                with self.ctrl_lock:
                    self.q_target = pre_yield_q.copy()
                time.sleep(0.01)

        if safe_deploy:
            if self.prepare_hands_on_deploy:
                logger_mp.info("[safe_arm_deploy] Phase 0: preparing hands...")
                _try_open_hands(duration=0.5)
            else:
                logger_mp.info(
                    "[safe_arm_deploy] Phase 0: skipped; external hand "
                    "controller is already active."
                )
            after_via_callback_done = False
            if self.safe_deploy_via_q is not None:
                logger_mp.info(
                    "[safe_arm_deploy] Phase 1a: moving through outward "
                    "clearance waypoint..."
                )
                self._move_dual_arm_waypoint(
                    self.safe_deploy_via_q,
                    "safe deploy clearance waypoint",
                    timeout=self.safe_deploy_via_min_duration + 1.0,
                    min_duration=self.safe_deploy_via_min_duration,
                    settle=False,
                )
                if callable(self.safe_deploy_after_via_callback):
                    logger_mp.info(
                        "[safe_arm_deploy] Phase 1a done: running "
                        "after-via callback..."
                    )
                    self.safe_deploy_after_via_callback()
                    after_via_callback_done = True
            logger_mp.info("[safe_arm_deploy] Phase 1b: moving to deploy target...")
            self._move_dual_arm_waypoint(
                self.safe_deploy_q,
                "safe deploy final target",
                timeout=self.safe_deploy_min_duration + 1.0,
                min_duration=self.safe_deploy_min_duration,
                settle=False,
            )
            if (
                callable(self.safe_deploy_after_via_callback)
                and not after_via_callback_done
            ):
                logger_mp.info(
                    "[safe_arm_deploy] Phase 1b done: running "
                    "after-via callback..."
                )
                self.safe_deploy_after_via_callback()
            if self.keep_spread:
                logger_mp.info(
                    "[safe_arm_deploy] keep_spread=True → staying at deploy target "
                    f"{np.round(self.safe_deploy_q, 3).tolist()}"
                )
            else:
                logger_mp.info("[safe_arm_deploy] Phase 2: moving to home q=0...")
                with self.ctrl_lock:
                    self.q_target = np.zeros(14)
                time.sleep(1.0)
            logger_mp.info("[safe_arm_deploy] Done.")

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].temperature = tuple(msg.motor_state[id].temperature)
                    lowstate.motor_state[id].mode = msg.motor_state[id].mode
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)

        # Soft boundary damping: decelerate as joints approach limits
        _SOFT_MARGIN = 0.15  # rad — damping zone width
        for i in range(14):
            dist_lo = cliped_arm_q_target[i] - _G1_29_ARM_Q_LOWER[i]
            dist_hi = _G1_29_ARM_Q_UPPER[i] - cliped_arm_q_target[i]
            nearest = min(dist_lo, dist_hi)
            if nearest < _SOFT_MARGIN:
                alpha = max(nearest, 0.0) / _SOFT_MARGIN
                cliped_arm_q_target[i] = alpha * cliped_arm_q_target[i] + (1.0 - alpha) * current_q[i]

        cliped_arm_q_target = np.clip(cliped_arm_q_target, _G1_29_ARM_Q_LOWER, _G1_29_ARM_Q_UPPER)

        for i in _G1_29_DISABLED_ARM_JOINTS:
            cliped_arm_q_target[i] = 0.0

        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0;

        disabled_motors = set()
        _last_temp_log = 0.0
        _last_wrist_diag_log = 0.0

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            grav = {}
            low_data = self.lowstate_buffer.GetData()
            if self.grav_comp.available and low_data is not None:
                grav = self.grav_comp.compute(
                    lambda u_idx: low_data.motor_state[u_idx].q
                )

            # Temperature safety check (every 2 seconds)
            if low_data is not None and (start_time - _last_temp_log) > 2.0:
                _last_temp_log = start_time
                for idx, id in enumerate(G1_29_JointArmIndex):
                    ms = low_data.motor_state[id]
                    max_temp = max(ms.temperature) if ms.temperature else 0
                    if max_temp >= _MOTOR_TEMP_DISABLE and id.value not in disabled_motors:
                        disabled_motors.add(id.value)
                        logger_mp.error(
                            f"[SAFETY] Motor {id.name}({id.value}) DISABLED — "
                            f"temp={list(ms.temperature)}, mode={ms.mode}. "
                            f"Holding current q to prevent damage."
                        )
                    elif max_temp >= _MOTOR_TEMP_WARN and id.value not in disabled_motors:
                        logger_mp.warning(
                            f"[SAFETY] Motor {id.name}({id.value}) HOT — "
                            f"temp={list(ms.temperature)}"
                        )
                    elif max_temp < _MOTOR_TEMP_WARN and id.value in disabled_motors:
                        disabled_motors.discard(id.value)
                        logger_mp.info(
                            f"[SAFETY] Motor {id.name}({id.value}) cooled down — "
                            f"temp={list(ms.temperature)}, re-enabled."
                        )

            for idx, id in enumerate(G1_29_JointArmIndex):
                if id.value in disabled_motors:
                    if low_data is not None:
                        cliped_arm_q_target[idx] = low_data.motor_state[id].q
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = 0.0

            if low_data is not None and (start_time - _last_wrist_diag_log) > 1.0:
                _last_wrist_diag_log = start_time
                wrist_items = [
                    ("LWR", 4, G1_29_JointArmIndex.kLeftWristRoll),
                    ("LWP", 5, G1_29_JointArmIndex.kLeftWristPitch),
                    ("LWY", 6, G1_29_JointArmIndex.kLeftWristyaw),
                    ("RWR", 11, G1_29_JointArmIndex.kRightWristRoll),
                    ("RWP", 12, G1_29_JointArmIndex.kRightWristPitch),
                    ("RWY", 13, G1_29_JointArmIndex.kRightWristYaw),
                ]
                diag_parts = []
                right_err_max = 0.0
                for name, arm_idx, joint in wrist_items:
                    actual = low_data.motor_state[joint.value].q
                    target = cliped_arm_q_target[arm_idx]
                    err = target - actual
                    temp = max(low_data.motor_state[joint.value].temperature or (0,))
                    diag_parts.append(
                        f"{name}:t={target:.3f} q={actual:.3f} e={err:.3f} T={temp}"
                    )
                    if name.startswith("R"):
                        right_err_max = max(right_err_max, abs(err))
                msg = "[WRIST_DIAG] " + " | ".join(diag_parts)
                if right_err_max > 0.18:
                    logger_mp.warning(msg)
                else:
                    logger_mp.info(msg)

            for j in _WAIST_JOINTS:
                self.msg.motor_cmd[j].tau = grav.get(j, 0.0)

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])
    
    def _move_dual_arm_waypoint(self, target_q, label,
                                timeout=5.0, tolerance=0.12,
                                min_duration=3.0, settle=True):
        start_q = self.get_current_dual_arm_q()
        target_q = target_q.copy()
        update_dt = 0.01
        logger_mp.info(
            f"[G1_29_ArmController] go_home: {label} "
            f"(slow {min_duration:.1f}s)..."
        )
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            alpha = 1.0 if min_duration <= 0 else min(1.0, elapsed / min_duration)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            with self.ctrl_lock:
                self.q_target = (1.0 - alpha) * start_q + alpha * target_q
            if alpha >= 1.0:
                break
            time.sleep(update_dt)

        if settle:
            deadline = time.time() + max(0.0, timeout - min_duration)
            while time.time() < deadline:
                current_q = self.get_current_dual_arm_q()
                if np.max(np.abs(current_q - target_q)) < tolerance:
                    break
                time.sleep(update_dt)

    def ctrl_dual_arm_go_home(self, lower_to_zero: bool = None,
                              keep_holder_yield: bool = False,
                              skip_spread: bool = False,
                              clearance_path: bool = False,
                              spread_min_duration: float = 2.0,
                              spread_timeout: float = 4.0,
                              spread_settle: bool = True,
                              prepare_hands: bool = True,
                              hand_settle_time: float = 0.3,
                              skip_zero_waypoint: bool = False,
                              park_q = None,
                              park_via_q = None,
                              park_via_min_duration: float = None,
                              park_via_timeout: float = None):
        '''Park the arms safely on teleop exit.

        Default behavior (since 2026-04-27): open hands → spread outward →
        STAY at spread (kNotUsedJoint0 weight stays = 1.0). The companion
        utils/arm_idle_holder.py daemon takes over right after we exit.
        This avoids the Dex3-1 fingers getting crushed against the outer
        thighs by the FSM's q=0 default standing pose.

        Pass `lower_to_zero=True` (or set self.keep_spread=False) to fall
        back to the legacy behavior: spread → q=0 → ramp arm_sdk weight
        down to 0 (release control).  Only do that if you have first
        verified that the Dex3 hands cannot collide in the resulting pose
        (e.g. after a hand replacement / mechanical re-design).

        Pass `keep_holder_yield=True` together with lower_to_zero when the
        user explicitly wants the factory/default arm pose after teleop exit;
        otherwise arm_idle_holder will resume and spread the arms again.

        Pass `clearance_path=True` for relax-to-default through an outward
        arc when needed; if already at the safe outer pose, go directly to q=0.

        Pass `skip_zero_waypoint=True` to avoid commanding the forward-ish
        q=0 arm pose before releasing arm_sdk control. This is useful when
        q=0 brings Dex3 hands close to the body.

        Pass `park_q` to override the default full-spread parking target.
        Single-arm teleop uses this to park the active arm outward while
        keeping the inactive arm in its relaxed hold pose between episodes.

        Pass `park_via_q` to move through a clearance waypoint before
        `park_q`. This avoids sweeping directly from a relaxed/down pose to
        the forward q=0 start pose.

        Pass `park_via_min_duration` / `park_via_timeout` to tune the
        clearance waypoint timing separately from the final park timing.
        '''
        if lower_to_zero is None:
            lower_to_zero = not getattr(self, "keep_spread", True)
        park_target = (
            np.asarray(park_q, dtype=np.float64).copy()
            if park_q is not None
            else _SPREAD_Q.copy()
        )
        park_via_target = (
            np.asarray(park_via_q, dtype=np.float64).copy()
            if park_via_q is not None
            else None
        )
        if park_via_min_duration is None:
            park_via_min_duration = spread_min_duration
        if park_via_timeout is None:
            park_via_timeout = spread_timeout

        logger_mp.info(
            f"[G1_29_ArmController] ctrl_dual_arm_go_home "
            f"start  (lower_to_zero={lower_to_zero})..."
        )

        if prepare_hands:
            # Phase 0: make fingers compact first.
            logger_mp.info("[G1_29_ArmController] go_home: closing/releasing hands...")
            _try_open_hands(duration=0.5)
            time.sleep(hand_settle_time)

        if clearance_path and lower_to_zero:
            current_q = self.get_current_dual_arm_q()
            non_roll_idx = [i for i in range(14) if i not in (1, 8)]
            already_outer = (
                current_q[1] > 1.25 and current_q[8] < -1.25
                and np.max(np.abs(current_q[non_roll_idx])) < 0.35
            )
            if already_outer:
                logger_mp.info(
                    "[G1_29_ArmController] go_home: already at outer "
                    "clearance pose; skip extra upward/outward waypoint."
                )
            else:
                outer_default = _CLEARANCE_Q.copy()
                self._move_dual_arm_waypoint(
                    outer_default, "following outer clearance arc",
                    timeout=5.0, min_duration=5.0, settle=False
                )
        elif skip_spread:
            logger_mp.info("[G1_29_ArmController] go_home: skipping spread.")
        else:
            if park_via_target is not None:
                self._move_dual_arm_waypoint(
                    park_via_target, "moving through park clearance waypoint",
                    timeout=park_via_timeout,
                    min_duration=park_via_min_duration,
                    settle=spread_settle,
                )
            # Phase 1: park at the configured safe target.
            self._move_dual_arm_waypoint(
                park_target, "moving to park target",
                timeout=spread_timeout, min_duration=spread_min_duration,
                settle=spread_settle,
            )

        if not lower_to_zero:
            # New default: hand off to arm_idle_holder. Clear yield flag so
            # the holder's next loop tick takes over publishing rt/arm_sdk.
            logger_mp.info(
                "[G1_29_ArmController] go_home: keep_spread=True → "
                "leaving arms at spread + arm_sdk weight=1, releasing "
                "yield flag for arm_idle_holder."
            )
            if self.motion_mode:
                _clear_yield_flag()
                # Give the holder a few publish cycles to wake up.
                time.sleep(0.5)
            return

        if skip_zero_waypoint:
            logger_mp.info(
                "[G1_29_ArmController] go_home: skipping q=0 waypoint; "
                "releasing arm_sdk from current safe pose."
            )
        else:
            # Final phase — move to factory/default arm pose.
            self._move_dual_arm_waypoint(
                np.zeros(14), "moving to q=0", timeout=7.0, min_duration=7.0,
                settle=not clearance_path,
            )

        if self.motion_mode:
            logger_mp.info("[G1_29_ArmController] go_home: ramping down slowly...")
            for weight in np.linspace(1, 0, num=201):
                self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight
                time.sleep(0.02)
            logger_mp.info("[G1_29_ArmController] arm_sdk weight = 0, control released.")
            if keep_holder_yield:
                _keep_holder_yielding_until_next_teleop()
            else:
                _clear_yield_flag()

    def safe_arm_deploy(self, spread_time=2.0, home_time=2.0):
        """Move arms outward first (avoid body collision), then to home (q=0).

        Phase 1 – spread: shoulder roll opens outward while pitch stays ~0
        and elbows stay straight.  This clears the torso.
        Phase 2 – home: all joints → 0 (standard teleop start pose).
        """
        # 14-element arm array index mapping:
        #  [1] L_ShoulderRoll  [8] R_ShoulderRoll
        spread_q = np.zeros(14)
        spread_q[1] = 1.5    # left shoulder roll → outward
        spread_q[8] = -1.5   # right shoulder roll → outward (mirrored)

        logger_mp.info("[safe_arm_deploy] Phase 1: spreading arms outward...")
        with self.ctrl_lock:
            self.q_target = spread_q.copy()
        time.sleep(spread_time)

        logger_mp.info("[safe_arm_deploy] Phase 2: moving to home (q=0)...")
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        time.sleep(home_time)
        logger_mp.info("[safe_arm_deploy] Done — arms at home position.")

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class G1_23_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode

        logger_mp.info("Initialize G1_23_ArmController...")
        self.q_target = np.zeros(10)
        self.tauff_target = np.zeros(10)

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_23_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_23_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_23_JointArmIndex)
        for id in G1_23_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_23_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_23_LowState()
                for id in range(G1_23_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_23_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_23_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_23_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(10)
            # self.tauff_target = np.zeros(10)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_23_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_23_JointIndex.kLeftAnklePitch.value,
            G1_23_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_23_JointIndex.kLeftShoulderPitch.value,
            G1_23_JointIndex.kLeftShoulderRoll.value,
            G1_23_JointIndex.kLeftShoulderYaw.value,
            G1_23_JointIndex.kLeftElbow.value,
            # Right arm
            G1_23_JointIndex.kRightShoulderPitch.value,
            G1_23_JointIndex.kRightShoulderRoll.value,
            G1_23_JointIndex.kRightShoulderYaw.value,
            G1_23_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors

class G1_23_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26

class G1_23_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRollNotUsed = 13
    kWaistPitchNotUsed = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristyawNotUsed = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class H1_2_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode
        
        logger_mp.info("Initialize H1_2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None


        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H1_2_JointArmIndex)
        for id in H1_2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_2_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_2_LowState()
                for id in range(H1_2_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_2_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[H1_2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    
class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_JointIndex(IntEnum):
    # Left leg
    kLeftHipYaw = 0
    kLeftHipRoll = 1
    kLeftHipPitch = 2
    kLeftKnee = 3
    kLeftAnkle = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipYaw = 6
    kRightHipRoll = 7
    kRightHipPitch = 8
    kRightKnee = 9
    kRightAnkle = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12

    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

    kNotUsedJoint0 = 27
    kNotUsedJoint1 = 28
    kNotUsedJoint2 = 29
    kNotUsedJoint3 = 30
    kNotUsedJoint4 = 31
    kNotUsedJoint5 = 32
    kNotUsedJoint6 = 33
    kNotUsedJoint7 = 34

class H1_ArmController:
    def __init__(self, simulation_mode = False):
        self.simulation_mode = simulation_mode
        
        logger_mp.info("Initialize H1_ArmController...")
        self.q_target = np.zeros(8)
        self.tauff_target = np.zeros(8)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, go_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, go_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_ArmController] Subscribe dds ok.")

        # initialize h1's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_go_msg_dds__LowCmd_()
        self.msg.head[0] = 0xFE
        self.msg.head[1] = 0xEF
        self.msg.level_flag = 0xFF
        self.msg.gpio = 0

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        for id in H1_JointIndex:
            if self._Is_weak_motor(id):
                self.msg.motor_cmd[id].kp = self.kp_low
                self.msg.motor_cmd[id].kd = self.kd_low
                self.msg.motor_cmd[id].mode = 0x01
            else:
                self.msg.motor_cmd[id].kp = self.kp_high
                self.msg.motor_cmd[id].kd = self.kd_high
                self.msg.motor_cmd[id].mode = 0x0A
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_LowState()
                for id in range(H1_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(8)
            # self.tauff_target = np.zeros(8)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[H1_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_JointIndex.kLeftAnkle.value,
            H1_JointIndex.kRightAnkle.value,
            # Left arm
            H1_JointIndex.kLeftShoulderPitch.value,
            H1_JointIndex.kLeftShoulderRoll.value,
            H1_JointIndex.kLeftShoulderYaw.value,
            H1_JointIndex.kLeftElbow.value,
            # Right arm
            H1_JointIndex.kRightShoulderPitch.value,
            H1_JointIndex.kRightShoulderRoll.value,
            H1_JointIndex.kRightShoulderYaw.value,
            H1_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
class H1_JointArmIndex(IntEnum):
    # Unlike G1 and H1_2, the arm order in DDS messages for H1 is right then left. 
    # Therefore, the purpose of switching the order here is to maintain consistency with G1 and H1_2.
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

class H1_JointIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kWaistYaw = 6
    kLeftHipYaw = 7
    kRightHipYaw = 8
    kNotUsedJoint = 9
    kLeftAnkle = 10
    kRightAnkle = 11
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
    import pinocchio as pin

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation

    arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    arm = G1_29_ArmController(simulation_mode=True)
    # arm_ik = G1_23_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_23_ArmController()
    # arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False)
    # arm = H1_2_ArmController()
    # arm_ik = H1_ArmIK(Unit_Test = True, Visualization = True)
    # arm = H1_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)