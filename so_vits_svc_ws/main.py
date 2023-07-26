from __future__ import annotations

import os
import sys
import json
import time
import base64
import socket
import asyncio
import threading
from io import BytesIO
from pathlib import Path
from logging import getLogger
from collections.abc import Callable
from typing import Any, Dict, List, Tuple, Union, Literal, Callable, Optional

import torch
import librosa
import uvicorn
import soundfile
import PySimpleGUI as sg
from patch import patch
from so_vits_svc_fork import __version__
from so_vits_svc_fork.inference.core import Svc
from so_vits_svc_fork.utils import get_optimal_device
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

GUI_DEFAULT_PRESETS_PATH = Path(__file__).parent / "default_gui_presets.json"
GUI_PRESETS_PATH = Path("./user_gui_presets.json").absolute()

LOG = getLogger(__name__)
patch()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def clear(self):
        self.active_connections.clear()

    async def close_all(self):
        for connection in self.active_connections:
            await connection.close()
        self.active_connections.clear()

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


class WebSocketServer:
    host: str
    port: int
    app: FastAPI
    manager: ConnectionManager
    receive_handle: Optional[Callable]
    uvicorn_thread: Optional[threading.Thread]
    send_lst: List
    send_exit: bool
    send_thread: Optional[threading.Thread]

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.manager = ConnectionManager()
        self.uvicorn_thread = None
        self.send_lst = []
        self.send_exit = False
        self.send_thread = None
    
        @self.app.websocket("/infer/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                while True:
                    json_str = await websocket.receive_text()
                    data = json.loads(json_str, object_hook=self.obj_hook)
                    if self.receive_handle is not None:
                        self.receive_handle(data)
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)

    def start(self):
        self.server = uvicorn.Server(uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level='warning'))
        self.uvicorn_thread = threading.Thread(target=self.server.run)
        self.uvicorn_thread.start()
        self.send_exit = False
        self.send_thread = threading.Thread(target=self.send_forever)
        self.send_thread.start()
        asyncio.run(self.wait_for_started())

    async def wait_for_started(self):
        while not self.server.started:
            await asyncio.sleep(0.1)

    def stop(self):
        if self.uvicorn_thread is not None and self.send_thread is not None and self.uvicorn_thread.is_alive():
            self.send_exit = True
            self.server.should_exit = True
            while self.uvicorn_thread.is_alive() or self.send_thread.is_alive():
                time.sleep(0.1)

    def add_response(self, results: List) -> None:
        self.send_lst.append(json.dumps(results, default=lambda x: base64.b64encode(x.getvalue()).decode() if isinstance(x, BytesIO) else x))

    def send_forever(self) -> None:
        while not self.send_exit:
            if len(self.send_lst) > 0:
                asyncio.run(self.manager.broadcast(self.send_lst.pop(0)))
            else:
                time.sleep(0.1)
        asyncio.run(self.manager.clear())
    
    @staticmethod
    def obj_hook(_json: Dict) -> Dict:
        if 'file' in _json:
            _json['file'] = BytesIO(base64.b64decode(_json['file']))
        elif 'data' in _json:
            _json['data'] = json.loads(_json['data'])
        return _json


class InferThread(threading.Thread):
    # path config
    model_path: Path
    config_path: Path
    recursive: bool = False
    # svc config
    speaker: Union[int, str]
    cluster_model_path: Union[Path, None] = None
    transpose: int = 0
    auto_predict_f0: bool = False
    cluster_infer_ratio: float = 0
    noise_scale: float = 0.4
    f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio"
    # slice config
    db_thresh: int = -40
    pad_seconds: float = 0.5
    chunk_seconds: float = 0.5
    absolute_thresh: bool = False
    max_chunk_seconds: float = 40
    # device config
    device: Union[str, torch.device] = get_optimal_device()
    # work
    svc_model: Optional[Svc]
    need_exit: bool
    thread_exit: bool
    task_lst: List[Tuple[str, BytesIO]]
    send_handle: Optional[Callable] = None
    model_reload: bool = False

    def __init__(
        self,
        # paths
        model_path: Union[Path, str],
        config_path: Union[Path, str],
        # svc config
        speaker: str,
        cluster_model_path: Union[Path, str, None] = None,
        transpose: int = 0,
        auto_predict_f0: bool = False,
        cluster_infer_ratio: float = 0,
        noise_scale: float = 0.4,
        f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
        chunk_seconds: float = 0.5,
        absolute_thresh: bool = False,
        max_chunk_seconds: float = 40,
        # device config
        device: Union[str, torch.device] = get_optimal_device(),
        # work
        send_handle: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.speaker = speaker
        self.cluster_model_path = Path(cluster_model_path) if cluster_model_path else None
        self.transpose = transpose
        self.auto_predict_f0 = auto_predict_f0
        self.cluster_infer_ratio = cluster_infer_ratio
        self.noise_scale = noise_scale
        self.f0_method = f0_method
        self.db_thresh = db_thresh
        self.pad_seconds = pad_seconds
        self.chunk_seconds = chunk_seconds
        self.absolute_thresh = absolute_thresh
        self.max_chunk_seconds = max_chunk_seconds
        self.device = device
        self.svc_model = None
        self.need_exit = False
        self.thread_exit = False
        self.task_lst = []
        self.send_handle = send_handle

    def set_config(
        self,
        # paths
        model_path: Union[Path, str],
        config_path: Union[Path, str],
        # svc config
        speaker: str,
        cluster_model_path: Union[Path, str, None] = None,
        transpose: int = 0,
        auto_predict_f0: bool = False,
        cluster_infer_ratio: float = 0,
        noise_scale: float = 0.4,
        f0_method: Literal["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"] = "dio",
        # slice config
        db_thresh: int = -40,
        pad_seconds: float = 0.5,
        chunk_seconds: float = 0.5,
        absolute_thresh: bool = False,
        max_chunk_seconds: float = 40,
        # device config
        device: Union[str, torch.device] = get_optimal_device(),
    ) -> None:
        model_reload = False
        def check_and_set(attribute: str, value: Any) -> bool:
            if self.__getattribute__(attribute) != value:
                self.__setattr__(attribute, value)
                return True
            return False
        model_reload |= check_and_set("model_path", Path(model_path))
        model_reload |= check_and_set("config_path", Path(config_path))
        model_reload |= check_and_set("cluster_model_path", Path(cluster_model_path) if cluster_model_path else None)
        model_reload |= check_and_set("speaker", speaker)
        model_reload |= check_and_set("transpose", transpose)
        model_reload |= check_and_set("auto_predict_f0", auto_predict_f0)
        model_reload |= check_and_set("cluster_infer_ratio", cluster_infer_ratio)
        model_reload |= check_and_set("noise_scale", noise_scale)
        model_reload |= check_and_set("f0_method", f0_method)
        model_reload |= check_and_set("db_thresh", db_thresh)
        model_reload |= check_and_set("pad_seconds", pad_seconds)
        model_reload |= check_and_set("chunk_seconds", chunk_seconds)
        model_reload |= check_and_set("absolute_thresh", absolute_thresh)
        model_reload |= check_and_set("max_chunk_seconds", max_chunk_seconds)
        model_reload |= check_and_set("device", device)
        if not self.model_reload and model_reload:
            self.model_reload = True

    def run(self) -> None:
        if self.svc_model is None:
            self.svc_model = Svc(
                net_g_path=self.model_path.as_posix(),
                config_path=self.config_path.as_posix(),
                cluster_model_path=self.cluster_model_path.as_posix() if self.cluster_model_path else None,
                device=self.device,
            )
        while not self.need_exit:
            if self.model_reload:
                del self.svc_model
                torch.cuda.empty_cache()
                self.svc_model = Svc(
                    net_g_path=self.model_path.as_posix(),
                    config_path=self.config_path.as_posix(),
                    cluster_model_path=self.cluster_model_path.as_posix() if self.cluster_model_path else None,
                    device=self.device,
                )
                self.model_reload = False
            if len(self.task_lst) > 0:
                try:
                    name, file = self.task_lst.pop(0)
                    file.name = str(Path(name).with_suffix('.ogg'))
                    try:
                        audio, _ = librosa.load(file, sr=self.svc_model.target_sample)
                    except Exception as e:
                        LOG.error(f"Failed to load {name}")
                        LOG.exception(e)
                        continue
                    audio = self.svc_model.infer_silence(
                        audio,  # type: ignore
                        speaker=self.speaker,
                        transpose=self.transpose,
                        auto_predict_f0=self.auto_predict_f0,
                        cluster_infer_ratio=self.cluster_infer_ratio,
                        noise_scale=self.noise_scale,
                        f0_method=self.f0_method,
                        db_thresh=self.db_thresh,
                        pad_seconds=self.pad_seconds,
                        chunk_seconds=self.chunk_seconds,
                        absolute_thresh=self.absolute_thresh,
                        max_chunk_seconds=self.max_chunk_seconds,
                    )
                    out_file = BytesIO()
                    soundfile.write(out_file, audio, samplerate=self.svc_model.target_sample, format='OGG')
                    if self.send_handle is not None:
                        self.send_handle([{"type": "voice", "name": name, "file": out_file}])
                except Exception as e:
                    LOG.exception(e)
            else:
                time.sleep(0.1)
        self.thread_exit = True

    def stop(self) -> None:
        self.need_exit = True
        while not self.thread_exit and self.is_alive():
            time.sleep(0.1)
        if self.svc_model is not None:
            del self.svc_model
            self.svc_model = None
        torch.cuda.empty_cache()

    def add_task(self, tasks: List[Dict]) -> None:
        for t in tasks:
            _type = t.get('type')
            if _type is None:
                continue
            if _type == 'voice':
                name = t.get('name')
                file = t.get('file')
                if name is not None and file is not None:
                    self.task_lst.append([name, file])  # type: ignore
            elif _type == 'cmd':
                name = t.get('name')
                data = t.get('data')
                if data is not None and isinstance(data, str):
                    data = json.loads(data)
                if name == 'cancel':
                    self.task_lst.clear()


def load_presets() -> Dict:
    defaults = json.loads(GUI_DEFAULT_PRESETS_PATH.read_text("utf-8"))
    users = (
        json.loads(GUI_PRESETS_PATH.read_text("utf-8"))
        if GUI_PRESETS_PATH.exists()
        else {}
    )
    # prioriy: defaults > users
    # order: defaults -> users
    return {**defaults, **users, **defaults}

def add_preset(name: str, preset: dict) -> dict:
    presets = load_presets()
    presets[name] = preset
    with GUI_PRESETS_PATH.open("w") as f:
        json.dump(presets, f, indent=2)
    return load_presets()

def delete_preset(name: str) -> dict:
    presets = load_presets()
    if name in presets:
        del presets[name]
    else:
        LOG.warning(f"Cannot delete preset {name} because it does not exist.")
    with GUI_PRESETS_PATH.open("w") as f:
        json.dump(presets, f, indent=2)
    return load_presets()

def main():
    LOG.info(f"version: {__version__}")
    if sys.platform == 'win32':
        threading.stack_size(16777216)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    sg.theme_add_new(
        "Very Dark",
        {
            "BACKGROUND": "#111111",
            "TEXT": "#FFFFFF",
            "INPUT": "#444444",
            "TEXT_INPUT": "#FFFFFF",
            "SCROLL": "#333333",
            "BUTTON": ("white", "#112233"),
            "PROGRESS": ("#111111", "#333333"),
            "BORDER": 2,
            "SLIDER_DEPTH": 2,
            "PROGRESS_DEPTH": 2,
        },
    )
    sg.theme("Very Dark")

    model_candidates = list(sorted(Path("./logs/44k/").glob("G_*.pth")))

    frame_contents = {
        "Paths": [
            [
                sg.Text("Model path"),
                sg.Push(),
                sg.InputText(
                    key="model_path",
                    default_text=model_candidates[-1].absolute().as_posix()
                    if model_candidates
                    else "",
                    enable_events=True,
                ),
                sg.FileBrowse(
                    initial_folder=Path("./logs/44k/").absolute
                    if Path("./logs/44k/").exists()
                    else Path(".").absolute().as_posix(),
                    key="model_path_browse",
                    file_types=(
                        ("PyTorch", "G_*.pth G_*.pt"),
                        ("Pytorch", "*.pth *.pt"),
                    ),
                ),
            ],
            [
                sg.Text("Config path"),
                sg.Push(),
                sg.InputText(
                    key="config_path",
                    default_text=Path("./configs/44k/config.json").absolute().as_posix()
                    if Path("./configs/44k/config.json").exists()
                    else "",
                    enable_events=True,
                ),
                sg.FileBrowse(
                    initial_folder=Path("./configs/44k/").as_posix()
                    if Path("./configs/44k/").exists()
                    else Path(".").absolute().as_posix(),
                    key="config_path_browse",
                    file_types=(("JSON", "*.json"),),
                ),
            ],
            [
                sg.Text("Cluster model path (Optional)"),
                sg.Push(),
                sg.InputText(
                    key="cluster_model_path",
                    default_text=Path("./logs/44k/kmeans.pt").absolute().as_posix()
                    if Path("./logs/44k/kmeans.pt").exists()
                    else "",
                    enable_events=True,
                ),
                sg.FileBrowse(
                    initial_folder="./logs/44k/"
                    if Path("./logs/44k/").exists()
                    else ".",
                    key="cluster_model_path_browse",
                    file_types=(("PyTorch", "*.pt"), ("Pickle", "*.pt *.pth *.pkl")),
                ),
            ],
        ],
        "Common": [
            [
                sg.Text("Speaker"),
                sg.Push(),
                sg.Combo(values=[], key="speaker", size=(20, 1)),
            ],
            [
                sg.Text("Silence threshold"),
                sg.Push(),
                sg.Slider(
                    range=(-60.0, 0),
                    orientation="h",
                    key="silence_threshold",
                    resolution=0.1,
                ),
            ],
            [
                sg.Text(
                    "Pitch (12 = 1 octave)\n"
                    "ADJUST THIS based on your voice\n"
                    "when Auto predict F0 is turned off.",
                    size=(None, 4),
                ),
                sg.Push(),
                sg.Slider(
                    range=(-36, 36),
                    orientation="h",
                    key="transpose",
                    tick_interval=12,
                ),
            ],
            [
                sg.Checkbox(
                    key="auto_predict_f0",
                    text="Auto predict F0 (Pitch may become unstable when turned on in real-time inference.)",
                )
            ],
            [
                sg.Text("F0 prediction method"),
                sg.Push(),
                sg.Combo(
                    ["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
                    key="f0_method",
                ),
            ],
            [
                sg.Text("Cluster infer ratio"),
                sg.Push(),
                sg.Slider(
                    range=(0, 1.0),
                    orientation="h",
                    key="cluster_infer_ratio",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("Noise scale"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 1.0),
                    orientation="h",
                    key="noise_scale",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("Pad seconds"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 1.0),
                    orientation="h",
                    key="pad_seconds",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("Chunk seconds"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 3.0),
                    orientation="h",
                    key="chunk_seconds",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("Max chunk seconds (set lower if Out Of Memory, 0 to disable)"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 240.0),
                    orientation="h",
                    key="max_chunk_seconds",
                    resolution=1.0,
                ),
            ],
            [
                sg.Checkbox(
                    key="absolute_thresh",
                    text="Absolute threshold (ignored (True) in realtime inference)",
                )
            ],
        ],
        "Presets": [
            [
                sg.Text("Presets"),
                sg.Push(),
                sg.Combo(
                    key="presets",
                    values=list(load_presets().keys()),
                    size=(40, 1),
                    enable_events=True,
                ),
                sg.Button("Delete preset", key="delete_preset"),
            ],
            [
                sg.Text("Preset name"),
                sg.Stretch(),
                sg.InputText(key="preset_name", size=(26, 1)),
                sg.Button("Add current settings as a preset", key="add_preset"),
            ],
        ],
        "Server": [
            [
                sg.Text("Host"),
                sg.Stretch(),
                sg.InputText(key="server_host", default_text="127.0.0.1", size=(26, 1)),
                sg.Text("Port"),
                sg.Stretch(),
                sg.InputText(key="server_port", default_text="11451", size=(26, 1)),
            ],
        ],
    }

    # frames
    frames = {}
    for name, items in frame_contents.items():
        frame = sg.Frame(name, items)
        frame.expand_x = True
        frames[name] = [frame]

    bottoms = [
        [
            sg.Checkbox(
                key="use_gpu",
                default=get_optimal_device() != torch.device("cpu"),
                text="Use GPU"
                + (
                    " (not available; if your device has GPU, make sure you installed PyTorch with CUDA support)"
                    if get_optimal_device() == torch.device("cpu")
                    else ""
                ),
                disabled=get_optimal_device() == torch.device("cpu"),
            )
        ],
        [
            sg.Button("(Re)Start Server", key="start_server"),
            sg.Button("Stop Server", key="stop_server"),
            sg.Push(),
        ],
    ]
    column1 = sg.Column(
        [
            frames["Paths"],
            frames["Common"],
            frames["Presets"],
            frames["Server"],
        ]
        + bottoms,
        vertical_alignment="top",
    )
    # columns
    layout = [[column1]]
    # get screen size
    screen_width, screen_height = sg.Window.get_screen_size()
    if screen_height < 720:
        layout = [
            [
                sg.Column(
                    layout,
                    vertical_alignment="top",
                    scrollable=False,
                    expand_x=True,
                    expand_y=True,
                    vertical_scroll_only=True,
                    key="main_column",
                )
            ]
        ]
    window = sg.Window(
        f"so-vits-svc-ws v{__version__}",
        layout,
        grab_anywhere=True,
        finalize=True,
        scaling=1,
        font=("Yu Gothic UI", 11) if os.name == "nt" else None,
    )
    
    # make slider height smaller
    try:
        for v in window.element_list():
            if isinstance(v, sg.Slider):
                v.Widget.configure(sliderrelief="flat", width=10, sliderlength=20)  # type: ignore
    except Exception as e:
        LOG.exception(e)

    event, values = window.read(timeout=0.01)  # type: ignore

    def update_speaker() -> None:
        from so_vits_svc_fork import utils

        config_path = Path(values["config_path"])
        if config_path.exists() and config_path.is_file():
            hp = utils.get_hparams(values["config_path"])
            LOG.debug(f"Loaded config from {values['config_path']}")
            window["speaker"].update(
                values=list(hp.__dict__["spk"].keys()), set_to_index=0
            )
    
    PRESET_KEYS = [
        key
        for key in values.keys()
        if not any(exclude in key for exclude in ["preset", "browse"])
    ]

    def apply_preset(name: str) -> None:
        for key, value in load_presets()[name].items():
            if key in PRESET_KEYS:
                window[key].update(value)
                values[key] = value

    default_name = list(load_presets().keys())[0]
    apply_preset(default_name)
    window["presets"].update(default_name)
    del default_name

    websocket_server = WebSocketServer('localhost', 11451)
    infer_thread = None
    while True:
        event, values = window.read(200)  # type: ignore
        if event == sg.WIN_CLOSED:
            websocket_server.stop()
            if infer_thread is not None and infer_thread.is_alive():
                infer_thread.stop()
            break
        if not event == sg.EVENT_TIMEOUT:
            LOG.info(f"Event {event}, values {values}")
        window["transpose"].update(
            disabled=values["auto_predict_f0"],
            visible=not values["auto_predict_f0"],
        )
        if event == "add_preset":
            presets = add_preset(
                values["preset_name"], {key: values[key] for key in PRESET_KEYS}
            )
            window["presets"].update(values=list(presets.keys()))
        elif event == "delete_preset":
            presets = delete_preset(values["presets"])
            window["presets"].update(values=list(presets.keys()))
        elif event == "presets":
            apply_preset(values["presets"])
            update_speaker()
        elif event == "config_path":
            update_speaker()
        elif event == "start_server":
            if values["model_path"] == '' or values["config_path"] == '':
                LOG.error(f"model_path, config_path can't be empty")
                continue
            websocket_server.stop()
            if infer_thread is not None and infer_thread.is_alive():
                infer_thread.stop()
            infer_thread = InferThread(
                # paths
                model_path=Path(values["model_path"]),
                config_path=Path(values["config_path"]),
                # svc config
                speaker=values["speaker"],
                cluster_model_path=Path(values["cluster_model_path"])
                if values["cluster_model_path"]
                else None,
                transpose=values["transpose"],
                auto_predict_f0=values["auto_predict_f0"],
                cluster_infer_ratio=values["cluster_infer_ratio"],
                noise_scale=values["noise_scale"],
                f0_method=values["f0_method"],
                # slice config
                db_thresh=values["silence_threshold"],
                pad_seconds=values["pad_seconds"],
                chunk_seconds=values["chunk_seconds"],
                absolute_thresh=values["absolute_thresh"],
                max_chunk_seconds=values["max_chunk_seconds"],
                device="cpu"
                if not values["use_gpu"]
                else get_optimal_device(),
            )
            infer_thread.send_handle = websocket_server.add_response
            websocket_server.receive_handle = infer_thread.add_task
            infer_thread.start()
            host = values["server_host"]
            port = str(values["server_port"])
            try:
                socket.inet_aton(host)
                websocket_server.host = host
            except socket.error:
                LOG.error(f"invalid server host")
            if port.isdigit() and 0 < int(port) < 25565:
                websocket_server.port = int(port)
            else:
                LOG.error(f"invalid server port")
            websocket_server.start()
        elif event == "stop_server":
            websocket_server.stop()
            if infer_thread is not None and infer_thread.is_alive():
                infer_thread.stop()
        if infer_thread is not None and infer_thread.is_alive():
            infer_thread.set_config(
                # paths
                model_path=Path(values["model_path"]),
                config_path=Path(values["config_path"]),
                # svc config
                speaker=values["speaker"],
                cluster_model_path=Path(values["cluster_model_path"])
                if values["cluster_model_path"]
                else None,
                transpose=values["transpose"],
                auto_predict_f0=values["auto_predict_f0"],
                cluster_infer_ratio=values["cluster_infer_ratio"],
                noise_scale=values["noise_scale"],
                f0_method=values["f0_method"],
                # slice config
                db_thresh=values["silence_threshold"],
                pad_seconds=values["pad_seconds"],
                chunk_seconds=values["chunk_seconds"],
                absolute_thresh=values["absolute_thresh"],
                max_chunk_seconds=values["max_chunk_seconds"],
                # device config
                device="cpu"
                if not values["use_gpu"]
                else get_optimal_device(),
            )
    window.close()

if __name__ == '__main__':
    main()
