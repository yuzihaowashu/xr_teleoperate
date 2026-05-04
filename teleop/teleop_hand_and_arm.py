import time
import signal
import argparse
from multiprocessing import Value, Array, Lock
import threading
import queue
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

signal.signal(signal.SIGHUP, signal.SIG_IGN)

# --- Non-blocking TTS via background thread ---
_tts_queue: queue.Queue = queue.Queue()

def _tts_worker():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    except Exception:
        return
    while True:
        text = _tts_queue.get()
        if text is None:
            break
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str):
    """Queue a TTS message for non-blocking playback."""
    _tts_queue.put_nowait(text)

import cv2
import numpy as np
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def draw_vr_hud(
    frame,
    tracking_active,
    recording,
    vr_connected,
    paused,
    loco_enabled=False,
    split_phase=None,
):
    """Overlay a minimal status HUD on the camera frame for VR display."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if recording:
        cv2.circle(overlay, (30, 30), 12, (0, 200, 0), -1)
        cv2.putText(overlay, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    elif tracking_active:
        cv2.circle(overlay, (30, 30), 12, (0, 200, 0), -1)
        cv2.putText(overlay, "READY", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    loco_label = "WALK ON" if loco_enabled else "WALK OFF"
    loco_color = (0, 200, 0) if loco_enabled else (128, 128, 128)
    cv2.putText(overlay, loco_label, (w - 160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, loco_color, 2)
    cv2.putText(
        overlay,
        "A SAVE | B DISCARD | Y TRANSITION",
        (12, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    if split_phase:
        phase = str(split_phase).upper()
        phase_color = (0, 0, 255) if split_phase == "forward" else (0, 200, 0)
        cv2.putText(
            overlay,
            phase,
            (130, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            phase_color,
            2,
        )

    if paused:
        label = "PAUSED — press X"
        cv2.putText(overlay, label, (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    elif not vr_connected:
        label = "VR DISCONNECTED"
        cv2.putText(overlay, label, (w // 2 - 160, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
LOCO_ENABLED   = False  # Locomotion disabled by default for safety
_EVENT_LOG     = []     # recent events for Gradio UI display
_EVENT_LOG_MAX = 20

_ARM_MODE_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}
_SPREAD_ARM_Q = np.zeros(14)
_SPREAD_ARM_Q[1] = 1.5
_SPREAD_ARM_Q[8] = -1.5

# Single-arm ready pose: keep the active arm at the original xr-teleoperate
# q=0 forward/default start pose instead of the outward Dex3 clearance pose.
_FORWARD_READY_ARM_Q = np.zeros(14)

# Softer than `spread` (±1.5 shoulder roll): modest pitch/roll + bent elbows so the
# inactive arm reads more "down / relaxed along the torso" than factory q=0 stand,
# which on G1 can still look quite open or T-pose-like next to the VR deploy spread.
_RELAXED_ARM_Q = np.zeros(14)
_RELAXED_ARM_Q[0] = 0.45   # L_ShoulderPitch
_RELAXED_ARM_Q[1] = 0.35   # L_ShoulderRoll
_RELAXED_ARM_Q[3] = 0.85   # L_Elbow
_RELAXED_ARM_Q[7] = 0.45   # R_ShoulderPitch
_RELAXED_ARM_Q[8] = -0.35  # R_ShoulderRoll
_RELAXED_ARM_Q[10] = 0.85  # R_Elbow


def _normalize_cli_arm_mode(mode: str) -> str:
    """Strip whitespace and normalize unicode hyphen variants (Gradio / copy-paste)."""
    m = str(mode or "").strip()
    for ch in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        m = m.replace(ch, "-")
    return m


def _inactive_arm_side(arm_mode: str):
    if arm_mode == "left-only":
        return "right"
    if arm_mode == "right-only":
        return "left"
    return None


def _active_arm_sides(arm_mode: str):
    if arm_mode == "left-only":
        return ("left",)
    if arm_mode == "right-only":
        return ("right",)
    return ("left", "right")


def _zero_like(values):
    return [0.0] * len(values)


def _rotation_matrix_to_quat(rot):
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    r = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-9)


def _quat_to_rotation_matrix(quat):
    """Convert a normalized xyzw quaternion to a 3x3 rotation matrix."""
    x, y, z, w = quat / max(np.linalg.norm(quat), 1e-9)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _slerp_rotation(prev_rot, target_rot, alpha):
    """Low-pass filter SO(3) with quaternion slerp."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha >= 1.0:
        return target_rot
    if alpha <= 0.0:
        return prev_rot

    q0 = _rotation_matrix_to_quat(prev_rot)
    q1 = _rotation_matrix_to_quat(target_rot)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
    else:
        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        q = s0 * q0 + s1 * q1
    return _quat_to_rotation_matrix(q)


def _select_inactive_arm_pose(pose_name: str, current_lr_arm_q):
    if pose_name == "current":
        return current_lr_arm_q.copy()
    if pose_name == "spread":
        return _SPREAD_ARM_Q.copy()
    if pose_name == "relaxed":
        return _RELAXED_ARM_Q.copy()
    return np.zeros(14, dtype=np.float64)


def _safe_deploy_q_for_arm_mode(arm_mode: str, inactive_pose: str):
    """Launch/park target: active side ready forward, inactive side held."""
    inactive_side = _inactive_arm_side(arm_mode)
    if inactive_side is None:
        return _SPREAD_ARM_Q.copy()

    deploy_q = _FORWARD_READY_ARM_Q.copy()
    inactive_q = _select_inactive_arm_pose(inactive_pose, deploy_q)
    side_slice = _ARM_MODE_SLICES[inactive_side]
    deploy_q[side_slice] = inactive_q[side_slice]
    return deploy_q


def _safe_deploy_via_q_for_arm_mode(arm_mode: str, inactive_pose: str):
    """Clearance waypoint: active side outward, inactive side held."""
    inactive_side = _inactive_arm_side(arm_mode)
    if inactive_side is None:
        return None

    via_q = _SPREAD_ARM_Q.copy()
    inactive_q = _select_inactive_arm_pose(inactive_pose, via_q)
    side_slice = _ARM_MODE_SLICES[inactive_side]
    via_q[side_slice] = inactive_q[side_slice]
    return via_q


def push_event(msg: str):
    """Log an event for both TTS and the Gradio event feed."""
    _EVENT_LOG.append(f"{time.strftime('%H:%M:%S')} — {msg}")
    if len(_EVENT_LOG) > _EVENT_LOG_MAX:
        del _EVENT_LOG[:-_EVENT_LOG_MAX]
    speak(msg)
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, LOCO_ENABLED
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
        push_event("Stop teleoperation")
        logger_mp.info("[keyboard] q pressed — STOP=True")
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'm':
        LOCO_ENABLED = not LOCO_ENABLED
        state_str = "ON" if LOCO_ENABLED else "OFF"
        push_event(f"Locomotion {state_str}")
        logger_mp.info(f"[keyboard] m pressed — LOCO_ENABLED={LOCO_ENABLED}")
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY, LOCO_ENABLED
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "LOCO_ENABLED": LOCO_ENABLED,
        "ARM_MODE": getattr(get_state, "arm_mode", "bimanual"),
        "EVENTS": list(_EVENT_LOG),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm-mode', type=str, choices=['bimanual', 'left-only', 'right-only'], default='bimanual',
                        help='Select which arm follows XR. In single-arm modes the inactive arm is held at --inactive-arm-pose and recorded with zero action.')
    parser.add_argument('--inactive-arm-pose', type=str, choices=['default', 'relaxed', 'spread', 'current'], default='default',
                        help='Hold pose for the inactive arm in single-arm modes: default=q=0 stand; relaxed=softer bent-elbow pose (less winged than spread); spread=outward shoulder roll for Dex3 clearance; current=snapshot at teleop start.')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--retarget-type', type=str, choices=['dexpilot', 'vector'], default='dexpilot', help='Hand retargeting algorithm for dex3 hand tracking mode (dexpilot=default, vector=Psi0-style)')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument(
        '--motion',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Keep Unitree motion/balance mode active and publish arms through rt/arm_sdk. Use --no-motion only for debug/bench tests.',
    )
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--auto-start-on-vr', action='store_true', help='Start tracking automatically once fresh VR data is detected')
    parser.add_argument('--auto-start-stable-sec', type=float, default=2.0, help='Seconds of continuous fresh VR data required before auto-start')
    parser.add_argument('--mirror-vr', action=argparse.BooleanOptionalAction, default=True, help='Show the VR camera/HUD mirror on the PC monitor')
    parser.add_argument('--force-zmq-video', action='store_true', help='Disable PC2 WebRTC and stream camera frames through the local TeleVuer page')
    parser.add_argument('--vr-pose-jump-threshold', type=float, default=0.15,
                        help='Pause controller tracking if one-frame wrist target jump exceeds this distance in meters')
    parser.add_argument('--vr-pose-filter-alpha', type=float, default=0.35,
                        help='Low-pass filter alpha for accepted controller wrist targets')
    parser.add_argument('--vr-rot-filter-alpha', type=float, default=1.0,
                        help='Low-pass filter alpha for accepted controller wrist rotations')
    parser.add_argument('--ik-rotation-weight', type=float, default=None,
                        help='IK wrist orientation tracking weight. Default: 1.0.')
    parser.add_argument('--wrist-kp', type=float, default=60.0,
                        help='G1_29 wrist joint position gain. Lower is smoother/quieter near the hand.')
    parser.add_argument('--wrist-kd', type=float, default=2.0,
                        help='G1_29 wrist joint damping gain. Tune with --wrist-kp for wrist smoothness.')
    parser.add_argument('--teleop-start-ramp-sec', type=float, default=2.5,
                        help='Seconds to blend from current arm joints to first IK targets after start/resume')
    parser.add_argument('--park-arms-on-stop', choices=['spread', 'default'], default='spread',
                        help='Final teleop stop pose: spread keeps Dex3 hands away from thighs; default releases to the factory arm pose and pauses arm_idle_holder.')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    args.arm_mode = _normalize_cli_arm_mode(args.arm_mode)
    get_state.arm_mode = args.arm_mode
    logger_mp.info(f"args: {args}")

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        if args.force_zmq_video:
            camera_config['head_camera']['enable_webrtc'] = False
            logger_mp.info("[video] force_zmq_video=True -> using host ZMQ frames for PICO display")
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])
        pc_mirror_enabled = args.mirror_vr and not args.headless and camera_config['head_camera']['enable_zmq']
        pc_mirror_state = {"enabled": pc_mirror_enabled, "warned": False}

        def show_pc_mirror(frame):
            """Best-effort PC observer window; never crash teleop if OpenCV has no GUI backend."""
            if not pc_mirror_state["enabled"]:
                return
            try:
                cv2.imshow("VR Mirror", frame)
                cv2.waitKey(1)
            except cv2.error as exc:
                pc_mirror_state["enabled"] = False
                if not pc_mirror_state["warned"]:
                    pc_mirror_state["warned"] = True
                    logger_mp.warning(
                        f"[VR Mirror] disabled because OpenCV GUI is unavailable: {exc}"
                    )

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # Motion mode keeps Unitree's balance/locomotion controller alive and
        # sends arm commands through rt/arm_sdk. Debug mode releases that
        # controller and is only for bench/debug use.
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.warning(
                f"Enter debug mode: {'Success' if status == 0 else 'Failed'}; "
                "leg balance/motion controller may be released."
            )

        # Start Dex3 controller before arm safe-deploy so fingers are held
        # continuously during the arm controller handoff.
        hand_ctrl = None
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            # Keep Dex3 compact during arm startup. Open after the arm first
            # reaches the outward clearance waypoint.
            left_trigger_value = Value('d', 1.0) if args.input_mode == "controller" else None
            right_trigger_value = Value('d', 1.0) if args.input_mode == "controller" else None
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                          left_trigger_value=left_trigger_value, right_trigger_value=right_trigger_value,
                                          retarget_type=args.retarget_type)

        # arm
        safe_deploy_q = None
        safe_deploy_via_q = None
        def _open_dex3_after_clearance():
            if args.ee == "dex3" and args.input_mode == "controller":
                left_trigger_value.value = 0.0
                right_trigger_value.value = 0.0
        if args.arm == "G1_29":
            if args.ik_rotation_weight is None:
                ik_rotation_weight = 1.0
            else:
                ik_rotation_weight = float(args.ik_rotation_weight)
            logger_mp.info(f"[IK] rotation_weight={ik_rotation_weight}")
            arm_ik = G1_29_ArmIK(rotation_weight=ik_rotation_weight)
            safe_deploy_q = _safe_deploy_q_for_arm_mode(args.arm_mode, args.inactive_arm_pose)
            safe_deploy_via_q = _safe_deploy_via_q_for_arm_mode(args.arm_mode, args.inactive_arm_pose)
            if args.arm_mode in ("left-only", "right-only"):
                logger_mp.info(
                    f"[ARM_MODE] {args.arm_mode}: launch deploy target "
                    f"{np.round(safe_deploy_q, 3).tolist()}"
                )
                logger_mp.info(
                    f"[ARM_MODE] {args.arm_mode}: launch via target "
                    f"{np.round(safe_deploy_via_q, 3).tolist()}"
                )
            arm_ctrl = G1_29_ArmController(
                motion_mode=args.motion,
                simulation_mode=args.sim,
                safe_deploy_q=safe_deploy_q,
                safe_deploy_via_q=safe_deploy_via_q,
                safe_deploy_via_min_duration=2.0,
                safe_deploy_min_duration=4.0,
                safe_deploy_after_via_callback=_open_dex3_after_clearance,
                prepare_hands_on_deploy=(args.ee != "dex3"),
                wrist_kp=args.wrist_kp,
                wrist_kd=args.wrist_kd,
            )
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3" and hand_ctrl is None:
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            left_trigger_value = Value('d', 0.0) if args.input_mode == "controller" else None
            right_trigger_value = Value('d', 0.0) if args.input_mode == "controller" else None
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                          left_trigger_value=left_trigger_value, right_trigger_value=right_trigger_value,
                                          retarget_type=args.retarget_type)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless,
                                     metadata = {
                                         "arm_mode": args.arm_mode,
                                         "inactive_arm_pose": args.inactive_arm_pose,
                                         "inactive_arm_action": "zero_delta",
                                     })

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            _con = Console()
            _tbl = Table(show_header=False, box=None, padding=(0, 2))
            _tbl.add_column(style="bold")
            _tbl.add_column()
            _tbl.add_row("[green]Keyboard [r][/]", "Start tracking")
            if args.auto_start_on_vr:
                _tbl.add_row("[green]PICO Enter VR[/]", "Auto-start tracking")
            _tbl.add_row("[green]VR Left X[/]", "Start/resume + record episode")
            _tbl.add_row("[green]VR Left Y[/]", "Mark transition: forward/backward")
            if args.record:
                _tbl.add_row("[yellow]VR Right A[/]", "Stop episode + save")
                _tbl.add_row("[red]VR Right B[/]", "Discard current episode")
                _tbl.add_row("[yellow]Keyboard [s][/]", "Toggle recording (manual)")
            if pc_mirror_state["enabled"]:
                _tbl.add_row("[blue]PC window[/]", "VR Mirror for audience")
            _tbl.add_row("[magenta]Keyboard [m][/]", "Toggle locomotion (default OFF)")
            _tbl.add_row("[red]Keyboard [q][/]", "Stop & exit teleop")
            _tbl.add_row("[cyan]Both joysticks pressed[/]", "Emergency damping")
            _con.print(Panel(_tbl, title="[bold]XR Teleoperate[/]",
                             subtitle=f"arm={args.arm}  arm_mode={args.arm_mode}  ee={args.ee}  input={args.input_mode}  record={'ON' if args.record else 'OFF'}  loco=OFF",
                             border_style="blue"))
            _con.print("[bold yellow]⚠  Keep safe distance from the robot![/]")
        except ImportError:
            logger_mp.info("Press [r] to start, [s] to toggle recording, [q] to quit.")
        READY = True                  # now ready to (1) enter START state
        _x_button_held = False
        _y_button_held = False
        _a_button_held = False
        _b_button_held = False
        _splitter_state = {
            "phase": "forward",
            "phase_index": 0,
            "transition_pending": False,
        }

        def _reset_splitter_phase(reason: str):
            _splitter_state["phase"] = "forward"
            _splitter_state["phase_index"] = 0
            _splitter_state["transition_pending"] = False
            logger_mp.info(f"[SPLITTER] reset phase to forward: {reason}")
        _VR_STALE_THRESHOLD = 2.0
        _vr_fresh_since = None
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and (xr_need_local_img or pc_mirror_state["enabled"]):
                head_img = img_client.get_head_frame()
                if head_img is not None and head_img.bgr is not None:
                    _frame = draw_vr_hud(
                        head_img.bgr.copy(),
                        False,
                        False,
                        True,
                        True,
                        LOCO_ENABLED,
                        _splitter_state["phase"],
                    )
                    if xr_need_local_img:
                        tv_wrapper.render_to_xr(_frame)
                    show_pc_mirror(_frame)
            _evt_time = tv_wrapper.last_event_time
            _vr_fresh = (_evt_time > 0) and (time.time() - _evt_time < _VR_STALE_THRESHOLD)
            if _vr_fresh:
                if _vr_fresh_since is None:
                    _vr_fresh_since = time.time()
            else:
                _vr_fresh_since = None
            if args.auto_start_on_vr and _vr_fresh_since is not None:
                stable_for = time.time() - _vr_fresh_since
                if stable_for >= args.auto_start_stable_sec:
                    START = True
                    logger_mp.info(
                        f"[VR] Fresh VR data stable for {stable_for:.1f}s -> auto-start tracking"
                    )
                    push_event("VR connected and stable. Auto-start tracking.")
                    break
            if args.input_mode == "controller" and hasattr(tv_wrapper, 'tvuer'):
                if tv_wrapper.tvuer.left_ctrl_aButton:
                    if not _x_button_held:
                        _x_button_held = True
                        START = True
                        logger_mp.info("[VR] Left X button pressed → start tracking")
                        if args.record:
                            RECORD_TOGGLE = True
                        _need_pose_guard_reset = True
                else:
                    _x_button_held = False

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        push_event("Start teleoperation")
        arm_ctrl.speed_gradual_max()
        _vr_connected = True
        _tracking_paused = False
        _pause_after_episode = False
        _need_pose_guard_reset = True
        _need_start_ramp_reset = True
        _pose_guard = {
            "left_prev": None,
            "right_prev": None,
            "left_filtered": None,
            "right_filtered": None,
        }
        _start_ramp = {
            "start_time": None,
            "start_q": None,
            "logged_done": False,
        }
        _inactive_side = _inactive_arm_side(args.arm_mode)
        _active_sides = _active_arm_sides(args.arm_mode)
        _inactive_arm_target = None
        if _inactive_side is not None:
            current_q = arm_ctrl.get_current_dual_arm_q()
            _inactive_arm_target = _select_inactive_arm_pose(args.inactive_arm_pose, current_q)
            side_slice = _ARM_MODE_SLICES[_inactive_side]
            logger_mp.info(
                f"[ARM_MODE] {args.arm_mode}: holding {_inactive_side} arm "
                f"at {args.inactive_arm_pose} pose {np.round(_inactive_arm_target[side_slice], 3).tolist()}"
            )
            push_event(f"Arm mode: {args.arm_mode}; {_inactive_side} arm held")

        def _reset_pose_guard(reason: str):
            _pose_guard["left_prev"] = None
            _pose_guard["right_prev"] = None
            _pose_guard["left_filtered"] = None
            _pose_guard["right_filtered"] = None
            logger_mp.info(f"[VR_POSE_GUARD] reset: {reason}")

        def _apply_pose_guard(tele_data):
            """Reject sudden OpenXR pose jumps and smooth accepted targets."""
            left_pos = tele_data.left_wrist_pose[:3, 3].copy()
            right_pos = tele_data.right_wrist_pose[:3, 3].copy()
            if _pose_guard["left_prev"] is None:
                _pose_guard["left_prev"] = left_pos
                _pose_guard["right_prev"] = right_pos
                _pose_guard["left_filtered"] = tele_data.left_wrist_pose.copy()
                _pose_guard["right_filtered"] = tele_data.right_wrist_pose.copy()
                logger_mp.info("[VR_POSE_GUARD] calibrated controller reference")
                return True

            left_jump = float(np.linalg.norm(left_pos - _pose_guard["left_prev"]))
            right_jump = float(np.linalg.norm(right_pos - _pose_guard["right_prev"]))
            jump_by_side = {"left": left_jump, "right": right_jump}
            jump = max(jump_by_side[side] for side in _active_sides)
            if jump > args.vr_pose_jump_threshold:
                logger_mp.warning(
                    f"[VR_POSE_GUARD] pose jump {jump:.3f}m "
                    f"(L={left_jump:.3f}, R={right_jump:.3f}, active={_active_sides})"
                )
                return False

            alpha = float(np.clip(args.vr_pose_filter_alpha, 0.0, 1.0))
            for side in _active_sides:
                pose = getattr(tele_data, f"{side}_wrist_pose")
                filt_key = f"{side}_filtered"
                filtered = _pose_guard[filt_key].copy()
                filtered[:3, 3] = (
                    (1.0 - alpha) * filtered[:3, 3] + alpha * pose[:3, 3]
                )
                rot_alpha = float(np.clip(args.vr_rot_filter_alpha, 0.0, 1.0))
                filtered[:3, :3] = _slerp_rotation(
                    filtered[:3, :3],
                    pose[:3, :3],
                    rot_alpha,
                )
                setattr(tele_data, f"{side}_wrist_pose", filtered)
                _pose_guard[filt_key] = filtered
                _pose_guard[f"{side}_prev"] = pose[:3, 3].copy()
            for side in set(("left", "right")) - set(_active_sides):
                pose = getattr(tele_data, f"{side}_wrist_pose")
                _pose_guard[f"{side}_prev"] = pose[:3, 3].copy()
            return True

        def _reset_start_ramp(reason: str):
            ramp_sec = max(0.0, float(args.teleop_start_ramp_sec))
            if ramp_sec <= 0:
                _start_ramp["start_time"] = None
                _start_ramp["start_q"] = None
                return
            _start_ramp["start_time"] = time.time()
            _start_ramp["start_q"] = arm_ctrl.get_current_dual_arm_q().copy()
            _start_ramp["logged_done"] = False
            logger_mp.info(f"[START_RAMP] reset: {reason}, duration={ramp_sec:.1f}s")
            push_event(f"Start ramp active ({ramp_sec:.1f}s)")

        def _apply_start_ramp(sol_q):
            if _start_ramp["start_time"] is None or _start_ramp["start_q"] is None:
                return sol_q
            ramp_sec = max(0.0, float(args.teleop_start_ramp_sec))
            elapsed = time.time() - _start_ramp["start_time"]
            if ramp_sec <= 0 or elapsed >= ramp_sec:
                if not _start_ramp["logged_done"]:
                    logger_mp.info("[START_RAMP] complete")
                    _start_ramp["logged_done"] = True
                _start_ramp["start_time"] = None
                _start_ramp["start_q"] = None
                return sol_q
            alpha = elapsed / ramp_sec
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            return (1.0 - alpha) * _start_ramp["start_q"] + alpha * sol_q

        def _apply_arm_mode(sol_q, sol_tauff):
            """Keep inactive arm out of teleop while preserving 14D command shape."""
            if _inactive_side is None:
                return sol_q, sol_tauff
            side_slice = _ARM_MODE_SLICES[_inactive_side]
            sol_q = sol_q.copy()
            sol_tauff = sol_tauff.copy()
            sol_q[side_slice] = _inactive_arm_target[side_slice]
            sol_tauff[side_slice] = 0.0
            return sol_q, sol_tauff

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()

            # --- Right A: finish current episode; keyboard q / panel stop exits teleop ---
            if args.input_mode == "controller" and hasattr(tv_wrapper, 'tvuer'):
                try:
                    if tv_wrapper.tvuer.right_ctrl_aButton:
                        if not _a_button_held:
                            _a_button_held = True
                            if args.motion:
                                loco_wrapper.Move(0, 0, 0)
                            if args.record:
                                if RECORD_RUNNING:
                                    RECORD_TOGGLE = True
                                    _pause_after_episode = True
                                    push_event("Stop episode. Saving...")
                                    logger_mp.info("[VR] Right A button pressed -> stop episode and save")
                                else:
                                    push_event("No active episode. Press X to start recording.")
                            else:
                                START = False
                                STOP = True
                                push_event("Stop teleoperation")
                                continue
                    else:
                        _a_button_held = False
                except Exception:
                    pass

            # --- VR connection monitoring ---
            _evt_time = tv_wrapper.last_event_time
            _vr_fresh = (_evt_time > 0) and (time.time() - _evt_time < _VR_STALE_THRESHOLD)
            if _vr_connected and not _vr_fresh:
                _vr_connected = False
                _tracking_paused = True
                if args.motion and args.input_mode == "controller":
                    loco_wrapper.Move(0, 0, 0)
                logger_mp.warning("[VR] Connection lost — tracking paused")
                push_event("VR disconnected. Tracking paused.")
            elif not _vr_connected and _vr_fresh:
                _vr_connected = True
                arm_ctrl.speed_gradual_max()
                logger_mp.info("[VR] Connection restored — press X to resume")
                push_event("VR reconnected. Press X to resume.")

            if _tracking_paused:
                if args.input_mode == "controller" and hasattr(tv_wrapper, 'tvuer'):
                    if tv_wrapper.tvuer.left_ctrl_aButton:
                        if not _x_button_held:
                            _x_button_held = True
                            _tracking_paused = False
                            START = True
                            _need_pose_guard_reset = True
                            _need_start_ramp_reset = True
                            if args.record and not RECORD_RUNNING:
                                RECORD_TOGGLE = True
                                push_event("Start next episode")
                            logger_mp.info("[VR] Resumed tracking after reconnection")
                            push_event("Tracking resumed. Controller pose recalibrated.")
                    else:
                        _x_button_held = False
                if camera_config['head_camera']['enable_zmq'] and (xr_need_local_img or pc_mirror_state["enabled"]):
                    head_img = img_client.get_head_frame()
                    if head_img is not None and head_img.bgr is not None:
                        _frame = draw_vr_hud(
                            head_img.bgr.copy(),
                            False,
                            False,
                            _vr_connected,
                            True,
                            LOCO_ENABLED,
                            _splitter_state["phase"],
                        )
                        if xr_need_local_img:
                            tv_wrapper.render_to_xr(_frame)
                        show_pc_mirror(_frame)
                time.sleep(0.033)
                continue

            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img or pc_mirror_state["enabled"]:
                    head_img = img_client.get_head_frame()
                if (xr_need_local_img or pc_mirror_state["enabled"]) and head_img is not None and head_img.bgr is not None:
                    _frame = draw_vr_hud(
                        head_img.bgr.copy(),
                        True,
                        RECORD_RUNNING,
                        _vr_connected,
                        False,
                        LOCO_ENABLED,
                        _splitter_state["phase"],
                    )
                    if xr_need_local_img:
                        tv_wrapper.render_to_xr(_frame)
                    show_pc_mirror(_frame)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        _reset_splitter_phase("new episode")
                        RECORD_RUNNING = True
                        push_event(f"Start recording episode {recorder.episode_id:04d}")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                        push_event("Recording failed")
                else:
                    saved_episode_id = recorder.episode_id
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    push_event(f"Saving episode {saved_episode_id:04d}...")
                    _save_status = {"text": f"SAVING EPISODE {saved_episode_id:04d}..."}

                    def _keep_vr_alive():
                        try:
                            if camera_config['head_camera']['enable_zmq'] and (xr_need_local_img or pc_mirror_state["enabled"]):
                                _hf = img_client.get_head_frame()
                                if _hf is not None and _hf.bgr is not None:
                                    _fr = draw_vr_hud(
                                        _hf.bgr.copy(),
                                        True,
                                        False,
                                        _vr_connected,
                                        False,
                                        LOCO_ENABLED,
                                        _splitter_state["phase"],
                                    )
                                    h, w = _fr.shape[:2]
                                    cv2.putText(_fr, _save_status["text"], (max(20, w // 2 - 260), h // 2),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 3)
                                    if xr_need_local_img:
                                        tv_wrapper.render_to_xr(_fr)
                                    show_pc_mirror(_fr)
                        except Exception:
                            pass

                    _save_deadline = time.time() + 30
                    while not recorder.is_ready():
                        if time.time() > _save_deadline:
                            logger_mp.error("recorder.is_ready() timed out (30s)")
                            break
                        time.sleep(0.033)
                        _keep_vr_alive()

                    push_event(f"Episode {saved_episode_id:04d} saved. Returning arms home...")
                    _save_status["text"] = f"EPISODE {saved_episode_id:04d} SAVED"

                    _go_home_done = threading.Event()
                    def _do_go_home():
                        try:
                            arm_ctrl.ctrl_dual_arm_go_home(
                                park_via_min_duration=3.0,
                                park_via_timeout=4.5,
                                spread_min_duration=6.0,
                                spread_timeout=7.5,
                                prepare_hands=False,
                                park_q=safe_deploy_q,
                                park_via_q=safe_deploy_via_q,
                            )
                        except Exception as _e:
                            logger_mp.error(f"go_home failed: {_e}")
                        finally:
                            _go_home_done.set()
                    threading.Thread(target=_do_go_home, daemon=True).start()
                    while not _go_home_done.wait(timeout=0.033):
                        _keep_vr_alive()

                    push_event(f"Episode {saved_episode_id:04d} saved. Press X for next episode.")
                    _save_status["text"] = f"READY - PRESS X FOR NEXT EPISODE"
                    for _ in range(15):
                        time.sleep(0.033)
                        _keep_vr_alive()
                    if _pause_after_episode:
                        START = False
                        _tracking_paused = True
                        _pause_after_episode = False
                    _reset_splitter_phase("episode saved")
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)
                    if _tracking_paused:
                        time.sleep(0.033)
                        continue

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if args.input_mode == "controller":
                if tele_data.left_ctrl_bButton:
                    if not _y_button_held:
                        _y_button_held = True
                        _splitter_state["phase"] = (
                            "backward"
                            if _splitter_state["phase"] == "forward"
                            else "forward"
                        )
                        _splitter_state["phase_index"] += 1
                        _splitter_state["transition_pending"] = True
                        push_event(
                            f"Splitter phase: {_splitter_state['phase']}"
                        )
                        logger_mp.info(
                            "[VR] Left Y button pressed -> "
                            f"split_phase={_splitter_state['phase']}"
                        )
                else:
                    _y_button_held = False

                if _need_pose_guard_reset:
                    _reset_pose_guard("controller pose reset")
                    _need_pose_guard_reset = False
                if _need_start_ramp_reset:
                    _reset_start_ramp("controller pose reset")
                    _need_start_ramp_reset = False
                if not _apply_pose_guard(tele_data):
                    _tracking_paused = True
                    START = False
                    if args.motion:
                        loco_wrapper.Move(0, 0, 0)
                    push_event("VR pose jump detected. Press X to recalibrate.")
                    time.sleep(0.033)
                    continue
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex3" and args.input_mode == "controller":
                # PICO trigger reports high when released and low when pressed.
                # Dex3 controller expects 0=open, 1=closed.
                left_trigger_value.value = 1.0 - (tele_data.left_ctrl_triggerValue / 10.0)
                right_trigger_value.value = 1.0 - (tele_data.right_ctrl_triggerValue / 10.0)
                if args.arm_mode == "left-only":
                    right_trigger_value.value = 0.0
                elif args.arm_mode == "right-only":
                    left_trigger_value.value = 0.0
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
                if args.arm_mode == "left-only":
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = 0.0
                elif args.arm_mode == "right-only":
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = 0.0
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # VR Right B discards the active episode. Only Right A saves.
            if args.input_mode == "controller":
                if args.record and tele_data.right_ctrl_bButton:
                    if not _b_button_held:
                        _b_button_held = True
                        if RECORD_RUNNING:
                            discarded_episode_id = recorder.episode_id
                            RECORD_RUNNING = False
                            recorder.discard_episode()
                            _reset_splitter_phase("episode discarded")
                            START = False
                            _tracking_paused = True
                            push_event(
                                f"Episode {discarded_episode_id:04d} discarded. "
                                "Press X for next episode."
                            )
                            logger_mp.info(
                                "[VR] Right B button pressed -> discard "
                                f"episode {discarded_episode_id:04d}"
                            )
                            time.sleep(0.033)
                            continue
                        push_event("No active episode to discard.")
                        logger_mp.info(
                            "[VR] Right B button pressed but no episode active"
                        )
                else:
                    _b_button_held = False

            # high level locomotion control
            if args.input_mode == "controller" and args.motion:
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # locomotion: only send Move commands when LOCO_ENABLED
                if LOCO_ENABLED:
                    _deadzone = 0.15
                    _lx = tele_data.left_ctrl_thumbstickValue[0] if abs(tele_data.left_ctrl_thumbstickValue[0]) > _deadzone else 0.0
                    _ly = tele_data.left_ctrl_thumbstickValue[1] if abs(tele_data.left_ctrl_thumbstickValue[1]) > _deadzone else 0.0
                    _rx = tele_data.right_ctrl_thumbstickValue[0] if abs(tele_data.right_ctrl_thumbstickValue[0]) > _deadzone else 0.0
                    loco_wrapper.Move(-_ly * 0.3, -_lx * 0.3, -_rx * 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # Single-arm: replace inactive VR wrist target with FK(inactive hold joints + measured active joints).
            # Coupled dual-arm IK + its smoother otherwise track the idle controller and both arms appear to move.
            if (
                _inactive_side is not None
                and _inactive_arm_target is not None
                and hasattr(arm_ik, "ee_pose_from_arm_q")
            ):
                try:
                    q_merge = current_lr_arm_q.copy()
                    sl = _ARM_MODE_SLICES[_inactive_side]
                    q_merge[sl] = _inactive_arm_target[sl]
                    if _inactive_side == "right":
                        tele_data.right_wrist_pose = arm_ik.ee_pose_from_arm_q(q_merge, "right")
                    else:
                        tele_data.left_wrist_pose = arm_ik.ee_pose_from_arm_q(q_merge, "left")
                except Exception as _fk_exc:
                    logger_mp.warning(f"[ARM_MODE] inactive wrist FK override failed: {_fk_exc}")

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            if args.input_mode == "controller":
                sol_q = _apply_start_ramp(sol_q)
            sol_q, sol_tauff = _apply_arm_mode(sol_q, sol_tauff)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")

            # Monitor wrist joints: indices 4=L_WristRoll, 5=L_WristPitch, 6=L_WristYaw,
            #                                11=R_WristRoll, 12=R_WristPitch, 13=R_WristYaw
            if not hasattr(arm_ik, '_dbg_t') or (time_ik_end - arm_ik._dbg_t) > 1.0:
                arm_ik._dbg_t = time_ik_end
                import numpy as _np
                _wrist_idx = [4, 5, 6, 11, 12, 13]
                _wrist_names = ['LWR','LWP','LWY','RWR','RWP','RWY']
                _vals = ' '.join(f'{_wrist_names[i]}={sol_q[_wrist_idx[i]]:.3f}' for i in range(6))
                logger_mp.info(f"[IK_DBG] {_vals}")

            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex3" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []
                if args.ee == "dex3" and hand_ctrl is not None:
                    pressure = hand_ctrl.get_current_dual_hand_pressure().tolist()
                    tactiles = {
                        "left_ee": pressure[:108],
                        "right_ee": pressure[108:],
                    }
                else:
                    tactiles = {
                        "left_ee": [],
                        "right_ee": [],
                    }

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if args.arm_mode == "left-only":
                    right_arm_action = np.zeros(7)
                    right_hand_action = _zero_like(right_hand_action)
                elif args.arm_mode == "right-only":
                    left_arm_action = np.zeros(7)
                    left_hand_action = _zero_like(left_hand_action)
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        },
                        "splitter": {
                            "phase": _splitter_state["phase"],
                            "phase_index": _splitter_state["phase_index"],
                            "transition": _splitter_state["transition_pending"],
                        },
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    _splitter_state["transition_pending"] = False
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactiles, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, tactiles=tactiles)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if args.motion and args.input_mode == "controller":
                loco_wrapper.Move(0, 0, 0)
                logger_mp.info("Sent Move(0,0,0) — locomotion stopped.")
        except Exception as e:
            logger_mp.error(f"Failed to stop locomotion: {e}")

        _end_effector_closed = {"done": False}
        def _close_end_effector_controller():
            if _end_effector_closed["done"]:
                return
            try:
                if args.ee == "dex3":
                    from teleop.robot_control.robot_hand_unitree import dex3_close_hands
                    hand_ctrl.close(release=False)
                    dex3_close_hands(duration=0.5, kp=0.45, kd=0.15)
                elif args.ee in ("inspire_dfx", "inspire_ftp", "brainco") and hasattr(hand_ctrl, "close"):
                    hand_ctrl.close()
                elif args.ee == "dex1" and hasattr(gripper_ctrl, "close"):
                    gripper_ctrl.close()
                _end_effector_closed["done"] = True
            except Exception as e:
                logger_mp.error(f"Failed to close end-effector controller: {e}")

        def _release_end_effector_controller():
            try:
                if args.ee == "dex3":
                    from teleop.robot_control.robot_hand_unitree import dex3_release_hands
                    dex3_release_hands(duration=0.6)
            except Exception as e:
                logger_mp.error(f"Failed to release end-effector controller: {e}")

        def _render_stop_overlay(text="STOPPING..."):
            """Render a status overlay to VR during shutdown."""
            try:
                if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    _hf = img_client.get_head_frame()
                    if _hf is not None and _hf.bgr is not None:
                        _fr = _hf.bgr.copy()
                    else:
                        _fr = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    _fr = np.zeros((480, 640, 3), dtype=np.uint8)
                h, w = _fr.shape[:2]
                cv2.putText(_fr, text, (w // 2 - 180, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
                tv_wrapper.render_to_xr(_fr)
                show_pc_mirror(_fr)
            except Exception:
                pass

        _render_stop_overlay("STOPPING...")
        push_event("Stopping — closing fingers safely...")
        _close_end_effector_controller()
        time.sleep(0.2)
        push_event("Stopping — arms returning home...")

        _go_home_done = threading.Event()
        def _do_exit_go_home():
            try:
                if args.arm == "G1_29":
                    final_stop_park_q = safe_deploy_q
                    final_stop_via_q = safe_deploy_via_q
                    if args.park_arms_on_stop == "spread":
                        # Final Stop Teleop should be simple: stretch, then
                        # Gradio's recovery relax step can lower/release arms.
                        # Do not visit the single-arm forward ready pose here.
                        final_stop_park_q = _SPREAD_ARM_Q.copy()
                        final_stop_via_q = None
                    arm_ctrl.ctrl_dual_arm_go_home(
                        lower_to_zero=(args.park_arms_on_stop == "default"),
                        keep_holder_yield=(args.park_arms_on_stop == "default"),
                        prepare_hands=False,
                        park_via_min_duration=2.0,
                        park_via_timeout=3.0,
                        spread_min_duration=4.0,
                        spread_timeout=5.0,
                        spread_settle=False,
                        park_q=final_stop_park_q,
                        park_via_q=final_stop_via_q,
                    )
                else:
                    arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as e:
                logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
            finally:
                _go_home_done.set()
        threading.Thread(target=_do_exit_go_home, daemon=True).start()
        while not _go_home_done.wait(timeout=0.033):
            _render_stop_overlay("STOPPING...")

        _render_stop_overlay("STOPPED")
        _release_end_effector_controller()
        push_event("Teleoperation ended")
        time.sleep(1.0)

        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        _close_end_effector_controller()

        try:
            if not args.motion:
                pass
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        time.sleep(0.5)
        _tts_queue.put(None)
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)