from typing import Any, Optional

import numpy as np
from loguru import logger
import rtde_control
import rtde_receive
import rtde_io 
import asyncio
from serial.tools.list_ports_common import ListPortInfo
from scipy.spatial.transform import Rotation as R

from phosphobot.hardware.base import BaseRobot
from phosphobot.models import RobotConfigStatus
from phosphobot.models.lerobot_dataset import BaseRobotInfo, FeatureDetails
from phosphobot.hardware.utils.robotiq_gripper import RobotiqGripper


class UR5eHardware(BaseRobot):
    name: str = "ur5e"
    is_connected: bool
    is_moving: bool
    with_gripper: bool

    def __init__(self, ip: str = "192.168.1.10", **kwargs):
        super().__init__(**kwargs)
        self.name = "ur5e"
        self.is_connected = False
        self.is_moving = False
        self.with_gripper = True
        self.num_joints = 6
        self.robot_ip = ip
        # Interfaces will be created in connect()
        self.rtde_ctrl = None
        self.rtde_rec = None
        self.rtde_inout = None
        # Conservative defaults (joint space)
        self.speed = float(kwargs.get("speed", 0.5))  # rad/s for moveJ
        self.acc = float(kwargs.get("acc", 0.5))      # rad/s^2 for moveJ
        self.gripper = RobotiqGripper()
        self.gripper_speed = 255
        self.gripper_force = 255

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot in radians.
        """
        self._raise_if_not_connected()
        q = np.asarray(q_target_rad, dtype=float).tolist()
        self.rtde_ctrl.moveJ(q, self.speed, self.acc)  # rad/s, rad/s^2

    def get_info_for_dataset(self) -> BaseRobotInfo:
        """
        Generate information about the robot useful for the dataset.
        Return a BaseRobotInfo object. (see models.dataset.BaseRobotInfo)
        Dict returned is info.json file at initialization
        """

        if self.with_gripper:
            num_joints = self.num_joints
        else:
            num_joints = self.num_joints - 1

        return BaseRobotInfo(
            robot_type="ur5e",
            action=FeatureDetails(
                dtype="float32",
                shape=[num_joints],
                names=[f"motor_{i}" for i in range(num_joints)],
            ),
            observation_state=FeatureDetails(
                dtype="float32",
                shape=[num_joints],
                names=[f"motor_{i}" for i in range(num_joints)],
            ),
        )

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.
        This method should return the observation of the robot.
        Will be used to build an observation in a Step of an episode.
        Returns:
            - state: np.array state of the robot (7D)
            - joints_position: np.array joints position of the robot
        """
        self._raise_if_not_connected()
        joints_position = np.asarray(self.rtde_rec.getActualQ(), dtype=float)
        tcp_pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
        logger.debug(f"q: {joints_position}, tcp: {tcp_pose}")
        return joints_position, tcp_pose

    async def connect(self) -> None:
        """
        Initialize communication with the robot.

        This method is called after the __init__ method.

        raise: Exception if the setup fails. For example, if the robot is not plugged in.
            This Exception will be caught by the __init__ method.
        """
        # Create interfaces
        self.rtde_ctrl = rtde_control.RTDEControlInterface(self.robot_ip)
        self.rtde_rec  = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.rtde_inout = rtde_io.RTDEIOInterface(self.robot_ip)
        self.gripper.connect(self.robot_ip, 63352)
        self.gripper.activate()

        self.is_connected = True
        logger.info("UR5e RTDE connected.")

    def disconnect(self) -> None:
        """
        Close the connection to the robot.

        This method is called on __del__ to disconnect the robot.
        """
        try:
            if self.rtde_ctrl: self.rtde_ctrl.disconnect()
            if self.rtde_rec:  self.rtde_rec.disconnect()
            if self.rtde_inout: self.rtde_inout.disconnect()
        finally:
            self.is_connected = False
            logger.info("UR5e RTDE disconnected.")

    def init_config(self) -> None:
        """
        Initialize the robot configuration.
        """
        pass

    def enable_torque(self) -> None:
        """
        Enable the torque of the robot.
        """
        pass

    def disable_torque(self) -> None:
        """
        Disable the torque of the robot.
        """
        pass

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,  # cartesian np.array
        target_orientation_rad: Optional[np.ndarray],  # rad np.array
        speed_m_s: float = 0.05,
        acc_m_s2: float = 0.2,
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation.
        This method should be implemented by the robot class.
        """
        self._raise_if_not_connected()
        if target_orientation_rad is None:
            # Keep current orientation (UR expects rotation vector)
            current_pose = self.rtde_rec.getActualTCPPose()
            rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
        else:
            # Convert Euler radians (xyz) to rotation vector for UR
            rotvec = R.from_euler("xyz", target_orientation_rad).as_rotvec()
            rx, ry, rz = float(rotvec[0]), float(rotvec[1]), float(rotvec[2])

        x, y, z = [float(v) for v in target_position]
        pose = [x, y, z, rx, ry, rz]                   # meters / rotvec (rad)
        self.rtde_ctrl.moveL(pose, speed_m_s, acc_m_s2)

    async def move_robot_relative(
        self,
        target_position: np.ndarray,  # delta in meters, elements may be None
        target_orientation_rad: Optional[np.ndarray],  # delta Euler radians, elements may be None
        step_time_s: float = 0.05,
        acc_m_s2: float = 0.5,
    ) -> None:
        """
        Discrete relative motion using servoL.

        - Computes an absolute target: current TCP pose + delta
        - Commands a short servo step (no background thread, no manual stop)
        - Repeated calls while a key is held feel continuous
        """
        self._raise_if_not_connected()

        # Read current pose (x,y,z, rotvec)
        curr = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
        pos = curr[:3]
        rotvec = curr[3:6]

        # Position delta (meters)
        dx, dy, dz = [float(v) if v is not None else 0.0 for v in target_position]
        new_pos = pos + np.array([dx, dy, dz], dtype=float)

        # Orientation composition: R_new = R_curr * R_delta
        R_curr = R.from_rotvec(rotvec)
        if target_orientation_rad is not None:
            d_euler = np.array([
                float(v) if (v is not None and not np.isnan(v)) else 0.0
                for v in target_orientation_rad
            ], dtype=float)
            R_delta = R.from_euler("xyz", d_euler)
            R_new = R_curr * R_delta
        else:
            R_new = R_curr

        new_rotvec = R_new.as_rotvec()

        # Clamp the step time to a small, stable range
        t = float(np.clip(step_time_s, 0.02, 0.1))

        # Conservative speed limit for the servo step
        v_lin = 0.3  # m/s
        a_lin = float(np.clip(acc_m_s2, 0.2, 1.0))

        pose = [
            float(new_pos[0]),
            float(new_pos[1]),
            float(new_pos[2]),
            float(new_rotvec[0]),
            float(new_rotvec[1]),
            float(new_rotvec[2]),
        ]

        # servoL(pose, a, v, t=0, lookahead_time=0.1, gain=300)
        # Use small t so a single call performs a short step and then stops
        self.rtde_ctrl.servoL(pose, a_lin, v_lin, t, 0.1, 300)

    def control_gripper(self, open_command: float) -> None:
        """
        Control the gripper.
        """

        open_command = int(open_command * 255)
        open_command = np.clip(open_command, 0, 255)
        try: 
            self.gripper.move(open_command, self.gripper_speed, self.gripper_force)
        except Exception as e:
            logger.error(f"Error controlling gripper: {e}")

    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["BaseRobot"]:
        """
        Return the robot class from the port information.
        """
        logger.warning(
            f"For automatic detection of {cls.__name__}, the method from_port must be implemented. Skipping autodetection."
        )
        return None

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(
            name=self.name,
            device_name=getattr(self, "SERIAL_ID", None),
        )

    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.
        The initial position is a safe position for the robot, where it is moved before starting the calibration.
        This method should be implemented by the robot class.

        This should update self.initial_position  and self.initial_orientation_rad
        """
        self._raise_if_not_connected()
        q_home = [-0.079, -1.98, 2.03, 3.70, -1.58, -4.78]
        self.rtde_ctrl.moveJ(q_home, 0.4, 0.6)
        self.gripper.move_and_wait_for_pos(self.gripper.get_open_position(), self.gripper_speed, self.gripper_force)
        # Record initial TCP pose for control zeroing
        pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
        position = pose[:3]
        rotvec = pose[3:6]
        euler_xyz = R.from_rotvec(rotvec).as_euler("xyz")
        self.initial_position = position
        self.initial_orientation_rad = euler_xyz

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.
        The sleep position is a safe position for the robot, where it is moved before disabling the motors.
        This method should be implemented by the robot class.
        """
        self._raise_if_not_connected()
        q_home = [-0.079, -1.98, 2.03, 3.70, -1.58, -4.78]
        self.rtde_ctrl.moveJ(q_home, 0.4, 0.6)

    def move_to_sleep_sync(self):
        asyncio.run(self.move_to_sleep())

    def _raise_if_not_connected(self):
        if not self.is_connected:
            raise Exception("Robot is not connected")

    def forward_kinematics(self, sync_robot_pos: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the end-effector position (meters) and orientation (Euler xyz radians)
        using the UR controller's current TCP pose. The sync_robot_pos argument is
        accepted for API compatibility and ignored for UR RTDE.
        """
        self._raise_if_not_connected()
        pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=float)
        position = pose[:3]
        rotvec = pose[3:6]
        euler_xyz = R.from_rotvec(rotvec).as_euler("xyz")
        return position, euler_xyz