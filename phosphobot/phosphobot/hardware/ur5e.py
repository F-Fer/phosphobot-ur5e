from typing import Any, Optional

import numpy as np
from loguru import logger
import rtde_control
import rtde_receive
import rtde_io 
import asyncio
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.hardware.base import BaseRobot
from phosphobot.models import RobotConfigStatus
from phosphobot.models.lerobot_dataset import BaseRobotInfo, FeatureDetails



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
        self.with_gripper = False
        self.num_joints = 6
        self.robot_ip = ip
        # Interfaces will be created in connect()
        self.rtde_ctrl = None
        self.rtde_rec = None
        self.rtde_inout = None
        # Conservative defaults (joint space)
        self.speed = float(kwargs.get("speed", 0.5))  # rad/s for moveJ
        self.acc = float(kwargs.get("acc", 0.5))      # rad/s^2 for moveJ

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
            # Keep current orientation
            current_pose = self.rtde_rec.getActualTCPPose()
            rx, ry, rz = current_pose[3], current_pose[4], current_pose[5]
        else:
            rx, ry, rz = [float(x) for x in target_orientation_rad]

        x, y, z = [float(v) for v in target_position]
        pose = [x, y, z, rx, ry, rz]                   # meters / radians
        self.rtde_ctrl.moveL(pose, speed_m_s, acc_m_s2)

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