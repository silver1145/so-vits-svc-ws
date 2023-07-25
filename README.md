# SO-VITS-SVC-WS

Added WebSocket server interaction method to [so-vits-svc-fork](https://github.com/voicepaw/so-vits-svc-fork).

## Install

### 1. From Hugging Face

Download embed package [here](https://huggingface.co/silver1145/SVC-WS/tree/main)

* Only for windows (>=10) and requires CUDA 11 (If CUDA is not installed, CPU mode also works)
* Run `start.bat` to start without log console

### 2. From Wheel

Create and activate a Python 3.10 virtual environment
install torch

```shell
pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Download [release wheel](https://github.com/silver1145/so-vits-svc-ws/releases)
Install the wheel

```shell
pip install so_vits_svc_ws-*.whl
```

start

```shell
svc-ws
```

### 3. Use Poetry Install
