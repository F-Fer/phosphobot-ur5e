from typing import Any, Optional, Literal

from dynamixel_sdk import (
    COMM_SUCCESS,
    PacketHandler,
    PortHandler,
)

import numpy as np
from loguru import logger
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.hardware.base import BaseManipulator
from phosphobot.utils import get_resources_path

class GelloUR(BaseManipulator):
    name = "gello-ur"

    URDF_FILE_PATH = str(get_resources_path() / "urdf" / "ur5e" / "urdf" / "ur5e.urdf")

    AXIS_ORIENTATION = [0, 0, 1, 1]

    END_EFFECTOR_LINK_INDEX = 5
    GRIPPER_JOINT_INDEX = 6 # Index 6 is the gripper joint (from 0 to 6)

    SERVO_IDS = [1, 2, 3, 4, 5, 6, 7]

    BAUDRATE = 57600
    RESOLUTION = 4096
    # (0*np.pi/2, 4*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2)
    CALIBRATION_POSITION = [0, 4*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2]

    # Control table addresses
    ADDR_PRESENT_POSITION = 132
    GRIPPER_ADDR_PRESENT_POSITION = 195
    GRIPPER_ADDR_GOAL_POSITION = 152

    calibration_max_steps = 1

    async def connect(self) -> None:

        # Initialize the port handler, packet handler, and group sync read/write
        self.portHandler = PortHandler(self.device_name)
        self.packetHandler = PacketHandler(protocol_version=2.0)

        # Open the port and set the baudrate
        if not self.portHandler.openPort():
            logger.warning("Failed to open the port")
            raise Exception("Failed to open the port")

        if not self.portHandler.setBaudRate(self.BAUDRATE):
            logger.warning("Failed to set the baud rate")
            raise Exception("Failed to set the baud rate")

        self.is_connected = True
        # Load calibration/config so unit conversions work
        self.init_config()

    def disconnect(self) -> None:
        if self.portHandler.is_open:
            self.portHandler.closePort()
        self.is_connected = False

    async def move_to_initial_position(self, open_gripper: bool = False) -> None:
        """
        Leader arm: do not command motors. Just set initial pose in sim.
        """
        # Ensure config is loaded for conversions elsewhere
        self.init_config()
        # Set initial pose to current simulated pose
        (
            self.initial_position,
            self.initial_orientation_rad,
        ) = self.forward_kinematics()

    def disable_torque(self) -> None:
        """
        This is a leader arm only. It does not have torque.
        """

        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return

        # logger.warning("GelloUR: This is a leader arm only. It does not have torque.")

    def enable_torque(self) -> None:
        """
        This is a leader arm only. It does not have torque.
        """

        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return

        # logger.warning("GelloUR: This is a leader arm only. It does not have torque.")

    def read_motor_torque(self, servo_id: int) -> Optional[float]:
        """
        Read the torque of a motor.
        """
        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return
        
        logger.warning("GelloUR: This is a leader arm only. Cannot read torque.")

    def read_motor_voltage(self, servo_id: int) -> Optional[float]:
        """
        Read the voltage of a motor.
        """
        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return
        
        logger.warning("GelloUR: This is a leader arm only. Cannot read voltage.")

    def write_motor_position(self, servo_id: int, units: int, **kwargs: Any) -> None:
        """
        Write the position of a motor.
        """
        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return
        
        logger.warning("GelloUR: This is a leader arm only. It is not possible to write the position of a motor.")

    def read_motor_position(self, servo_id: int, **kwargs: Any) -> Optional[int]:
        """
        Read the position of a motor.
        """
        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return
        
        try:
            (
                position,
                dxl_comm_result,
                dxl_error,
            ) = self.packetHandler.read4ByteTxRx(
                self.portHandler, servo_id, self.ADDR_PRESENT_POSITION
            )
            if dxl_comm_result != COMM_SUCCESS:
                logger.warning(
                    f"Communication Error for motor {servo_id}: {self.packetHandler.getTxRxResult(dxl_comm_result)}"
                )
            elif dxl_error != 0:
                logger.warning(
                    f"Hardware Error for motor {servo_id}: {self.packetHandler.getRxPacketError(dxl_error)}"
                )
            else:
                # sign correction for 32-bit two's complement
                if position > 0x7FFFFFFF:
                    position -= 0x100000000
                return position
        except Exception as e:
            logger.error(f"Error reading present position for motor {servo_id}: {e}")

        return None

    def calibrate_motors(self, **kwargs: Any) -> None:
        """
        Calibrate the motors.
        """
        return None

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        """
        One-shot calibration for the leader arm:
        - Reads current motor units and stores them as servos_offsets (defines 0 rad pose)
        - Keeps existing signs and calibration positions if available, else from defaults
        - Saves per-serial config to ~/.phosphobot/calibration/
        """

        if not self.is_connected:
            self.calibration_current_step = 0
            logger.warning(
                "Robot is not connected. Cannot calibrate. Calibration sequence reset to 0."
            )
            return (
                "error",
                "Robot is not connected. Cannot calibrate. Calibration sequence reset to 0.",
            )

        # Read current joint positions in motor units
        try:
            current_units = self.read_joints_position(unit="motor_units", source="robot")
            print(f"Current units: {current_units}")
        except Exception as e:
            logger.error(f"Failed reading joints for calibration: {e}")
            return ("error", f"Failed reading joints: {e}")

        if current_units is None or any(np.isnan(current_units)):
            return ("error", "Invalid joint readings (None/NaN). Check connection.")

        # Load existing/default config as baseline
        cfg = self.config
        if cfg is None:
            cfg = self.get_default_base_robot_config(voltage="6V")

        # Update offsets with current readings
        offsets = cfg.servos_offsets
        if len(offsets) != len(self.SERVO_IDS):
            offsets = [2048.0] * len(self.SERVO_IDS)
        for i in range(len(self.SERVO_IDS) - 1): # -1 because the last one is the gripper
            offsets[i] = float(current_units[i])
        offsets[len(self.SERVO_IDS) - 1] = float(current_units[len(self.SERVO_IDS) - 1] - 575) # 575 is the travel of the gripper
        cfg.servos_offsets = offsets

        # Ensure signs length and defaults
        signs = cfg.servos_offsets_signs
        if len(signs) != len(self.SERVO_IDS):
            # Use common GELLO signs as default if mismatched
            default_signs = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
            # Pad/truncate to servo count
            signs = (default_signs + [1.0] * len(self.SERVO_IDS))[: len(self.SERVO_IDS)]
        cfg.servos_offsets_signs = signs

        # Ensure calibration positions length
        cal_pos = cfg.servos_calibration_position
        if len(cal_pos) != len(self.SERVO_IDS):
            cal_pos = [2200.0] * len(self.SERVO_IDS)
        cal_pos[len(self.SERVO_IDS) - 1] = float(current_units[len(self.SERVO_IDS) - 1]) # Set the open position of the gripper
        cfg.servos_calibration_position = cal_pos

        # Persist per-serial config
        # Ensure SERIAL_ID: if missing, derive from device_name basename
        serial_id = getattr(self, "SERIAL_ID", None)
        if not serial_id:
            device_basename = (self.device_name or "no_device").split("/")[-1]
            serial_id = device_basename.replace(" ", "_")
            self.SERIAL_ID = serial_id

        try:
            saved_path = cfg.save_local(serial_id=self.SERIAL_ID)
            self.config = cfg
            print(f"Calibration saved to {saved_path}")
        except Exception as e:
            logger.error(f"Failed saving calibration: {e}")
            return ("error", f"Failed saving calibration: {e}")

        self.calibration_current_step = self.calibration_max_steps
        return ("success", f"Calibration saved to {saved_path}")

if __name__ == "__main__":
    import asyncio
    import time
    gello_ur = GelloUR(device_name="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMO0B-if00-port0")
    asyncio.run(gello_ur.connect())
    asyncio.run(gello_ur.calibrate())
    while True:
        joints_position = gello_ur.read_joints_position(unit="motor_units", source="robot")
        # Parse to int
        joints_position = [int(position) for position in joints_position]
        print(joints_position)
        time.sleep(0.5)