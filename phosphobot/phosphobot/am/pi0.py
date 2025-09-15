from phosphobot.am.base import ActionModel
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import logging
from phosphobot.models import ModelConfigurationResponse
from phosphobot.camera import AllCameras
from phosphobot.control_signal import AIControlSignal
from phosphobot.hardware.base import BaseManipulator
import time
from collections import deque
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    from openpi_client import websocket_client_policy  # type: ignore
    from openpi_client import image_tools

    class Pi0(ActionModel):
        def __init__(
            self,
            server_url: str = "http://127.0.0.1",
            server_port: int = 8000
        ):
            if server_url is None:
                logger.error("Server URL is not set. Please set the server URL.")
                raise ValueError("Server URL is not set. Please set the server URL.")
            if server_port is None:
                logger.error("Server port is not set. Please set the server port.")
                raise ValueError("Server port is not set. Please set the server port.")

            super().__init__(server_url, server_port)
            self.server_url = server_url
            self.server_port = server_port
            self.required_input_keys: List[str] = ["images", "state", "prompt"]
            self.image_keys = ["observation/exterior_image_1_left", "observation/wrist_image_left"]

            # Instantiate the client
            try:
                logger.info(f"Initializing Pi0 with server_url: {server_url} and server_port: {server_port}")
                self.client = websocket_client_policy.WebsocketClientPolicy(
                    host=self.server_url,
                    port=self.server_port,
                    api_key=None,
                )
                logger.info(f"Successfully connected to the server: {self.client.get_server_metadata()}")
            except Exception as e:
                logger.error(f"Error instantiating the client: {e}")
                raise e
            

        @classmethod
        def fetch_and_get_configuration(cls, model_id: str) -> ModelConfigurationResponse:
            """
            Fetch the model from Hugging Face and get the configuration.
            Args:
                model_id (str): Model ID on Hugging Face.
            Returns:
                video_keys, list[str]: List of configuration keys.
                checkpoints, list[str]: List of available checkpoints.
            """
            raise NotImplementedError(
                f"This method is not implemented in {cls.__name__}. You need to implement it in your subclass."
            )


        def sample_actions(self, inputs: dict) -> np.ndarray:
            observation = {
                "observation/joint_position": inputs["state"][:7],
                "observation/gripper_position": inputs["state"][-1],
                "prompt": inputs["prompt"]
            }

            # Map each configured image key to the corresponding image input
            for i in range(0, len(self.image_keys)):
                if i < len(inputs["images"]):
                    observation[self.image_keys[i]] = image_tools.convert_to_uint8(image_tools.resize_with_pad(inputs["images"][i], 224, 224))
                else:
                    logger.warning(f"Not enough images provided. Reusing the last available one. {len(inputs['images'])} images provided, {len(self.image_keys)} image keys expected.")
                    # If not enough images provided, reuse the last available one
                    observation[self.image_keys[i]] = image_tools.convert_to_uint8(image_tools.resize_with_pad(inputs["images"][-1], 224, 224))

            # Call the remote server
            try:
                action_chunk = self.client.infer(observation)["actions"]
            except Exception as e:
                logger.error(f"Error calling the remote server: {e}")
                raise e

            logger.debug(f"Action chunk type: {type(action_chunk)}")
            if type(action_chunk) == np.ndarray:
                logger.debug(f"Action chunk shape: {action_chunk.shape}")
            else:
                logger.debug(f"Action chunk: {action_chunk}")

            # TODO: check action_chunk is of type np.ndarray
            return action_chunk

        async def control_loop(
            self,
            control_signal: AIControlSignal,
            robots: List[BaseManipulator],
            model_spawn_config: Optional[object],
            all_cameras: AllCameras,
            fps: int = 30,
            speed: float = 1.0,
            cameras_keys_mapping: Optional[Dict[str, int]] = None,
            prompt: Optional[str] = None,
            selected_camera_id: Optional[int] = None,
            angle_format: Literal["degrees", "rad", "other"] = "rad",
            min_angle: Optional[float] = None,
            max_angle: Optional[float] = None,
            **kwargs: Any,
        ) -> None:
            """
            Control loop for a remote OpenPI policy.

            - Grabs frames from cameras according to self.image_keys or mapping
            - Reads concatenated robot joint states
            - Sends observation to remote policy and executes returned trajectory chunk
            """

            signal_marked_as_started = False
            actions_queue: deque = deque([])

            # Helper to resolve which camera id to use for each expected image key
            def resolve_camera_id(index: int, key: str) -> int:
                if cameras_keys_mapping is not None:
                    # Try exact key, then simplified variants
                    if key in cameras_keys_mapping:
                        return cameras_keys_mapping[key]
                    simplified = key.replace("observation/", "").replace("video.", "")
                    if simplified in cameras_keys_mapping:
                        return cameras_keys_mapping[simplified]
                    # Fallback to index
                    return cameras_keys_mapping.get(str(index), index)
                return index

            # Map angle_format to robot IO unit
            unit: Literal["rad", "motor_units", "degrees", "other"]
            if angle_format == "degrees":
                unit = "degrees"
            elif angle_format == "rad":
                unit = "rad"
            else:
                unit = "other"

            while control_signal.is_in_loop():
                start_time = time.perf_counter()

                # Build images list following configured keys
                images: List[np.ndarray] = []
                num_expected = max(1, len(self.image_keys))
                for i in range(num_expected):
                    key = self.image_keys[i]
                    cam_id = resolve_camera_id(i, key)
                    frame = all_cameras.get_rgb_frame(camera_id=cam_id)
                    if frame is None:
                        # Default to a black frame if camera not available
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    images.append(frame)

                # Read and concatenate robot joint states
                if len(robots) == 0:
                    control_signal.stop()
                    logger.error("No robot connected. Exiting AI control loop.")
                    break
                
                state = robots[0].get_observation()[1] # Get the joints positions (idx 1)
                for robot in robots[1:]:
                    state = np.concatenate(
                        (
                            state,
                            robot.get_observation()[1], # Get the joints positions (idx 1)
                        ),
                        axis=0,
                    )

                logger.debug(f"State: {state}")

                inputs: Dict[str, Any] = {
                    "images": images,
                    "state": state,
                    "prompt": prompt or "",
                }

                try:
                    if len(actions_queue) == 0:
                        actions = self.sample_actions(inputs)
                        logger.debug(f"Actions: {actions}")
                        # Normalize actions to shape (T, action_dim)
                        if not isinstance(actions, np.ndarray):
                            actions = np.array(actions)
                        if actions.ndim == 1:
                            actions = actions.reshape(1, -1)
                        actions_queue.extend(actions)

                    current_actions = actions_queue.popleft()
                    logger.debug(f"Current actions: {current_actions}")

                except Exception as e:
                    logger.warning(
                        f"Failed to get actions from remote policy: {e}. Exiting AI control loop."
                    )
                    control_signal.stop()
                    break

                if not signal_marked_as_started:
                    control_signal.set_running()
                    signal_marked_as_started = True

                # Dispatch actions to robots
                action_list = current_actions.tolist()
                for robot_index in range(len(robots)):
                    start = robot_index * 6
                    end = start + 6
                    target_position = action_list[start:end]
                    robots[robot_index].set_motors_positions(
                        q_target_rad=target_position,
                        enable_gripper=True
                    )

                # Pace the loop
                elapsed_time = time.perf_counter() - start_time
                sleep_time = max(0, 1.0 / (fps * speed) - elapsed_time)
                await asyncio.sleep(sleep_time)

except ImportError:

    class Pi0(ActionModel):  # type: ignore
        def __init__(self, server_url: str = "localhost", server_port: int = 8080):
            raise NotImplementedError(
                "Pi0 model requires openpi_client package: https://github.com/phospho-app/openpi.git"
            )
