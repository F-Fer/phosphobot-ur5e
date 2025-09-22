from typing import Any, Optional

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

    END_EFFECTOR_LINK_INDEX = 6
    GRIPPER_JOINT_INDEX = 7

    SERVO_IDS = [1, 2, 3, 4, 5, 6, 7]

    BAUDRATE = 57600
    RESOLUTION = 4096
    # (0*np.pi/2, 4*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2)
    CALIBRATION_POSITION = [0, 4*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2, 2*np.pi/2]

    # Control table addresses
    ADDR_PRESENT_POSITION = 132

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs: Any) -> Optional["GelloUR"]:
        """
        Detect if the device is a Gello UR leader robot.
        """
        # TODO: check if the device is a Gello UR leader robot.
        if port.pid == 21971 and port.serial_number in {"58CD176940"}:
            return cls(device_name=port.device, serial_id=port.serial_number)
        return None

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

    def disconnect(self) -> None:
        if self.portHandler.is_open:
            self.portHandler.closePort()
        self.is_connected = False

    def disable_torque(self) -> None:
        """
        This is a leader arm only. It does not have torque.
        """

        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return

        logger.warning("GelloUR: This is a leader arm only. It does not have torque.")

    def enable_torque(self) -> None:
        """
        This is a leader arm only. It does not have torque.
        """

        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return

        logger.warning("GelloUR: This is a leader arm only. It does not have torque.")

    def read_motor_torque(self, servo_id: int) -> Optional[float]:
        """
        Read the torque of a motor.
        """
        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return
        
        logger.warning("GelloUR: This is a leader arm only. It does not have torque.")

    def read_motor_voltage(self, servo_id: int) -> Optional[float]:
        """
        Read the voltage of a motor.
        """
        if not self.is_connected:
            logger.warning("GelloUR: Not connected. Run .connect() first.")
            return
        
        logger.warning("GelloUR: This is a leader arm only. It does not have voltage.")

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
                return position
        except Exception as e:
            logger.error(f"Error reading present position for motor {servo_id}: {e}")

        return None

    def calibrate_motors(self, **kwargs: Any) -> None:
        # TODO: implement
        return None

if __name__ == "__main__":
    import asyncio
    gello_ur = GelloUR(device_name="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAAMO0B-if00-port0")
    asyncio.run(gello_ur.connect())
    print("Connected to Gello UR")
    print(gello_ur.read_motor_position(1))
    gello_ur.disconnect()
    print("Disconnected from Gello UR")