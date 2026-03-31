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

def draw_vr_hud(frame, tracking_active, recording, vr_connected, paused, loco_enabled=False):
    """Overlay a minimal status HUD on the camera frame for VR display."""
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if recording:
        cv2.circle(overlay, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(overlay, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif tracking_active:
        cv2.circle(overlay, (30, 30), 12, (0, 200, 0), -1)
        cv2.putText(overlay, "READY", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    loco_label = "WALK ON" if loco_enabled else "WALK OFF"
    loco_color = (0, 200, 0) if loco_enabled else (128, 128, 128)
    cv2.putText(overlay, loco_label, (w - 160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, loco_color, 2)

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
        "EVENTS": list(_EVENT_LOG),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--retarget-type', type=str, choices=['dexpilot', 'vector'], default='dexpilot', help='Hand retargeting algorithm for dex3 hand tracking mode (dexpilot=default, vector=Psi0-style)')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
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
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

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
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
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
        if args.ee == "dex3":
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
                                     rerun_log = not args.headless)

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            _con = Console()
            _tbl = Table(show_header=False, box=None, padding=(0, 2))
            _tbl.add_column(style="bold")
            _tbl.add_column()
            _tbl.add_row("[green]Keyboard [r][/]", "Start tracking")
            _tbl.add_row("[green]VR Left X[/]", "Start tracking (controller mode)")
            if args.record:
                _tbl.add_row("[yellow]Keyboard [s] / VR Right B[/]", "Toggle recording")
            _tbl.add_row("[magenta]Keyboard [m][/]", "Toggle locomotion (default OFF)")
            _tbl.add_row("[red]Keyboard [q] / VR Right A[/]", "Stop & exit")
            _tbl.add_row("[cyan]Both joysticks pressed[/]", "Emergency damping")
            _con.print(Panel(_tbl, title="[bold]XR Teleoperate[/]",
                             subtitle=f"arm={args.arm}  ee={args.ee}  input={args.input_mode}  record={'ON' if args.record else 'OFF'}  loco=OFF",
                             border_style="blue"))
            _con.print("[bold yellow]⚠  Keep safe distance from the robot![/]")
        except ImportError:
            logger_mp.info("Press [r] to start, [s] to toggle recording, [q] to quit.")
        READY = True                  # now ready to (1) enter START state
        _x_button_held = False
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img is not None and head_img.bgr is not None:
                    _frame = draw_vr_hud(head_img.bgr.copy(), False, False, True, True, LOCO_ENABLED)
                    tv_wrapper.render_to_xr(_frame)
            if args.input_mode == "controller" and hasattr(tv_wrapper, 'tvuer'):
                if tv_wrapper.tvuer.left_ctrl_aButton:
                    if not _x_button_held:
                        _x_button_held = True
                        START = True
                        logger_mp.info("[VR] Left X button pressed → start tracking")
                else:
                    _x_button_held = False

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        push_event("Start teleoperation")
        arm_ctrl.speed_gradual_max()
        _b_button_held = False
        _vr_connected = True
        _VR_STALE_THRESHOLD = 2.0
        _tracking_paused = False
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()

            # --- A button (stop) — checked first so it ALWAYS works ---
            if args.input_mode == "controller" and hasattr(tv_wrapper, 'tvuer'):
                try:
                    if tv_wrapper.tvuer.right_ctrl_aButton:
                        if args.motion:
                            loco_wrapper.Move(0, 0, 0)
                        START = False
                        STOP = True
                        push_event("Stop teleoperation")
                        continue
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
                            logger_mp.info("[VR] Resumed tracking after reconnection")
                            push_event("Tracking resumed")
                    else:
                        _x_button_held = False
                if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    head_img = img_client.get_head_frame()
                    if head_img is not None and head_img.bgr is not None:
                        _frame = draw_vr_hud(head_img.bgr.copy(), False, False, _vr_connected, True, LOCO_ENABLED)
                        tv_wrapper.render_to_xr(_frame)
                time.sleep(0.033)
                continue

            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img is not None and head_img.bgr is not None:
                    _frame = draw_vr_hud(head_img.bgr.copy(), True, RECORD_RUNNING, _vr_connected, False, LOCO_ENABLED)
                    tv_wrapper.render_to_xr(_frame)
                    if not args.headless:
                        cv2.imshow("VR Mirror", _frame)
                        cv2.waitKey(1)
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
                        RECORD_RUNNING = True
                        push_event("Start recording")
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                        push_event("Recording failed")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    push_event("Saving episode...")

                    def _keep_vr_alive():
                        try:
                            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                                _hf = img_client.get_head_frame()
                                if _hf is not None and _hf.bgr is not None:
                                    _fr = draw_vr_hud(_hf.bgr.copy(), True, False, _vr_connected, False, LOCO_ENABLED)
                                    tv_wrapper.render_to_xr(_fr)
                                    if not args.headless:
                                        cv2.imshow("VR Mirror", _fr)
                                        cv2.waitKey(1)
                        except Exception:
                            pass

                    _save_deadline = time.time() + 30
                    while not recorder.is_ready():
                        if time.time() > _save_deadline:
                            logger_mp.error("recorder.is_ready() timed out (30s)")
                            break
                        time.sleep(0.033)
                        _keep_vr_alive()

                    push_event("Episode saved. Returning arms home...")

                    _go_home_done = threading.Event()
                    def _do_go_home():
                        try:
                            arm_ctrl.ctrl_dual_arm_go_home()
                        except Exception as _e:
                            logger_mp.error(f"go_home failed: {_e}")
                        finally:
                            _go_home_done.set()
                    threading.Thread(target=_do_go_home, daemon=True).start()
                    while not _go_home_done.wait(timeout=0.033):
                        _keep_vr_alive()

                    push_event("Ready for next episode.")
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex3" and args.input_mode == "controller":
                left_trigger_value.value = 1.0 - (tele_data.left_ctrl_triggerValue / 10.0)
                right_trigger_value.value = 1.0 - (tele_data.right_ctrl_triggerValue / 10.0)
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    loco_wrapper.Move(0, 0, 0)
                    START = False
                    STOP = True
                    push_event("Stop teleoperation")
                    continue
                # B button: toggle recording (same as keyboard [s])
                if args.record and tele_data.right_ctrl_bButton:
                    if not _b_button_held:
                        _b_button_held = True
                        RECORD_TOGGLE = True
                        logger_mp.info("[VR] Right B button pressed → toggle recording")
                else:
                    _b_button_held = False
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

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
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

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
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
                            colors[f"color_{0}"] = head_img
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
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

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
                if not args.headless:
                    cv2.imshow("VR Mirror", _fr)
                    cv2.waitKey(1)
            except Exception:
                pass

        _render_stop_overlay("STOPPING...")
        push_event("Stopping — arms returning home...")

        _go_home_done = threading.Event()
        def _do_exit_go_home():
            try:
                arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as e:
                logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
            finally:
                _go_home_done.set()
        threading.Thread(target=_do_exit_go_home, daemon=True).start()
        while not _go_home_done.wait(timeout=0.033):
            _render_stop_overlay("STOPPING...")

        _render_stop_overlay("STOPPED")
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