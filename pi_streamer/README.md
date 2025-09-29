# Pi Streamer

Lightweight ZeroMQ PUSH server that captures frames from USB cameras and sends
them to the Phosphobot workstation. Intended for deployment on a USB3-capable
edge device (e.g., Raspberry Pi 4, Jetson Nano) located near the cameras.

## Setup

1. Flash Raspberry Pi OS (64-bit recommended) and enable SSH.
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-opencv libatlas-base-dev ffmpeg
   pip3 install uv
   ```
3. Clone this repository and copy `pi_streamer/` to the device.
4. Create a virtual env using `uv`:
   ```bash
   cd pi_streamer
   uv sync
   ```
5. Copy `config.example.json` to `config.json` and adjust devices/topics.
6. Test locally:
   ```bash
   uv run python streamer.py --config config.json
   ```
7. On the Phosphobot workstation, use the Admin UI or POST `/cameras/add-zmq`
   to register each stream with the corresponding topic.

## Systemd Service (optional)

Create `/etc/systemd/system/pi-streamer.service`:

```
[Unit]
Description=Phosphobot Pi Streamer
After=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pi_streamer
ExecStart=/home/pi/.local/bin/uv run python streamer.py --config config.json
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Then enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pi-streamer
```

## Notes

- Ensure the edge device provides USB3 bandwidth and power for the ZED Mini.
- Tune `jpeg_quality` and `fps` to match your network throughput.
- Monitor `journalctl -u pi-streamer` for runtime diagnostics.


