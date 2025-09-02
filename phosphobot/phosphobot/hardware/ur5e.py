# phosphobot/phosphobot/hardware/ur5e.py
from phosphobot.hardware.base import BaseRobot
from ur_rtde import rtde_control, rtde_receive

class UR5eHardware(BaseRobot):
    NAME = "ur5e"
    DOF = 6
    HAS_GRIPPER = False
    URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "ur5e" / "urdf" / "ur5e.urdf"
    )

    def __init__(self, 
    only_simulation: bool = False,
    host: str = "192.168.0.2",
    speed: float = 0.3,
    acc: float = 0.6,
    **kwargs):
        super().__init__(**kwargs)
        self.only_simulation = only_simulation
        self.host = host
        self.speed = float(speed)
        self.acc   = float(acc)
        self.ctrl = None
        self.recv = None

    # required by BaseRobot
    def connect(self):
        self.ctrl = rtde_control.RTDEControlInterface(self.host)
        self.recv = rtde_receive.RTDEReceiveInterface(self.host)
        return True

    def disconnect(self):
        if self.ctrl: self.ctrl.disconnect()
        if self.recv: self.recv.disconnect()

    def get_joint_positions(self):
        return self.recv.getActualQ()

    def set_joint_positions(self, q):
        # blocking joint move; for continuous control you’ll implement servo/streamed variants
        self.ctrl.moveJ(q, self.speed, self.acc)