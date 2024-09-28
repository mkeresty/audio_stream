import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
import asyncio
import numpy as np
import threading
import time
import warnings
import yt_dlp
import subprocess
import io
import os
import psutil
from collections import deque
import datetime
from starlette.websockets import WebSocketState, WebSocketDisconnect
import whisper

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Audio recording settings
CHUNK = 1024
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# Flags to control behavior
USE_YT_DLP = True
SAVE_TO_FILE = False

BUFFER_SIZE = 10  # Number of audio chunks to buffer
CHUNK_DURATION = RECORD_SECONDS  # Duration of each chunk in seconds

# Load Whisper model
model = whisper.load_model("tiny")

class PerformanceMonitor:
    def __init__(self):
        self.transcription_count = 0
        self.total_transcription_time = 0
        self.start_time = time.time()
        self.process = psutil.Process()
        self.total_latency = 0

    def update_metrics(self, transcription_time, latency):
        self.transcription_count += 1
        self.total_transcription_time += transcription_time
        self.total_latency += latency

    def get_metrics(self):
        return {
            "avg_transcription_time": self.total_transcription_time / max(1, self.transcription_count),
            "avg_latency": self.total_latency / max(1, self.transcription_count),
            "memory_usage_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_usage_percent": self.process.cpu_percent(),
            "uptime_seconds": time.time() - self.start_time,
            "transcription_count": self.transcription_count
        }

performance_monitor = PerformanceMonitor()

class SharedTranscription:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.audio_generator = None
        self.clients = set()
        self.lock = asyncio.Lock()
        self.start_time = datetime.datetime.now()
        self.last_refresh = time.time()

    async def start(self):
        self.audio_generator = record_audio_yt_dlp(self.stream_url)

    async def stop(self):
        if self.audio_generator:
            await self.audio_generator.aclose()

    def get_info(self):
        return {
            "stream_url": self.stream_url,
            "client_count": len(self.clients),
            "uptime": str(datetime.datetime.now() - self.start_time)
        }

    async def process_audio(self):
        while True:
            try:
                async for audio_chunk, chunk_timestamp in self.audio_generator:
                    yield audio_chunk, chunk_timestamp
                    
                    # Refresh the stream every 30 seconds
                    if time.time() - self.last_refresh > 30:
                        self.last_refresh = time.time()
                        self.audio_generator = record_audio_yt_dlp(self.stream_url)
                        break
            except Exception as e:
                logger.error(f"Error in process_audio: {str(e)}")
                await asyncio.sleep(5)  # Wait a bit before restarting
                self.audio_generator = record_audio_yt_dlp(self.stream_url)

active_transcriptions = {}

async def get_latest_stream_url(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'no_color': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = await asyncio.to_thread(ydl.extract_info, youtube_url, download=False)
        if info is None:
            raise Exception("Failed to extract stream info")
        
        formats = info.get('formats', [info])
        audio_format = next((f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none'), None)
        if audio_format is None:
            raise Exception("No suitable audio format found")
        
        return audio_format['url']

async def record_audio_yt_dlp(youtube_url):
    while True:
        try:
            audio_url = await get_latest_stream_url(youtube_url)
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '5',
                '-i', audio_url,
                '-f', 's16le',
                '-ar', str(RATE),
                '-ac', str(CHANNELS),
                '-'
            ]

            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )

            start_time = time.time()
            while True:
                audio_chunk = await process.stdout.read(CHUNK * CHANNELS * 2 * RECORD_SECONDS)
                if not audio_chunk:
                    break
                
                yield audio_chunk, time.time()
                
                # Refresh the stream URL every 30 seconds
                if time.time() - start_time > 30:
                    break

            process.terminate()
            await process.wait()

        except Exception as e:
            logger.error(f"Error in record_audio_yt_dlp: {str(e)}")
            await asyncio.sleep(5)  # Wait a bit before retrying

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    stream_url = websocket.query_params.get("stream")
    if not stream_url:
        await websocket.close(code=1000, reason="No stream URL provided")
        return

    shared_transcription = None
    try:
        async with asyncio.Lock():
            if stream_url not in active_transcriptions:
                shared_transcription = SharedTranscription(stream_url)
                active_transcriptions[stream_url] = shared_transcription
                await shared_transcription.start()
            else:
                shared_transcription = active_transcriptions[stream_url]

        shared_transcription.clients.add(websocket)

        async for audio_chunk, chunk_timestamp in shared_transcription.process_audio():
            if websocket.client_state == WebSocketState.DISCONNECTED:
                break

            chunk_start_time = time.time()
            
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            transcription_start = time.time()
            result = model.transcribe(audio_np)
            transcription_time = time.time() - transcription_start
            
            latency = time.time() - chunk_start_time
            performance_monitor.update_metrics(transcription_time, latency)
            
            if result["text"]:
                timestamp = datetime.datetime.fromtimestamp(chunk_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                message = {
                    "type": "transcription",
                    "text": f"[{timestamp}] {result['text']}"
                }
                await asyncio.gather(*[client.send_json(message) for client in shared_transcription.clients if client.client_state == WebSocketState.CONNECTED])
                logger.info(f"Transcribed: [{timestamp}] {result['text']}")
            
            performance_message = {
                "type": "performance",
                **performance_monitor.get_metrics()
            }
            await asyncio.gather(*[client.send_json(performance_message) for client in shared_transcription.clients if client.client_state == WebSocketState.CONNECTED])

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket_endpoint: {str(e)}")
    finally:
        if shared_transcription:
            shared_transcription.clients.discard(websocket)
            if not shared_transcription.clients:
                await shared_transcription.stop()
                del active_transcriptions[stream_url]
        
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.get("/")
async def get(request: Request):
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>WebSocket Transcription</title>
        </head>
        <body>
            <h1>WebSocket Transcription</h1>
            <input type="text" id="stream_url" placeholder="Enter YouTube URL">
            <button onclick="connect()">Connect</button>
            <div id="transcription">
                <h2>Transcriptions</h2>
                <ul id='messages'></ul>
            </div>
            <div id='performance'>
                <h2>Performance Metrics</h2>
                <p>Average Transcription Time: <span id='avg_time'></span> seconds</p>
                <p>Average Latency: <span id='avg_latency'></span> seconds</p>
                <p>Memory Usage: <span id='memory'></span> MB</p>
                <p>CPU Usage: <span id='cpu'></span>%</p>
                <p>Uptime: <span id='uptime'></span> seconds</p>
                <p>Transcription Count: <span id='count'></span></p>
            </div>
            <script>
                var ws;
                function connect() {
                    var stream_url = document.getElementById('stream_url').value;
                    ws = new WebSocket(`ws://${location.host}/ws?stream=${encodeURIComponent(stream_url)}`);
                    ws.onmessage = function(event) {
                        var data = JSON.parse(event.data);
                        if (data.type === "transcription") {
                            var messages = document.getElementById('messages');
                            var message = document.createElement('li');
                            var content = document.createTextNode(data.text);
                            message.appendChild(content);
                            messages.appendChild(message);
                            messages.scrollTop = messages.scrollHeight;
                        } else if (data.type === "performance") {
                            document.getElementById('avg_time').textContent = data.avg_transcription_time.toFixed(3);
                            document.getElementById('avg_latency').textContent = data.avg_latency.toFixed(3);
                            document.getElementById('memory').textContent = data.memory_usage_mb.toFixed(2);
                            document.getElementById('cpu').textContent = data.cpu_usage_percent.toFixed(2);
                            document.getElementById('uptime').textContent = data.uptime_seconds.toFixed(0);
                            document.getElementById('count').textContent = data.transcription_count;
                        }
                    };
                    ws.onclose = function(event) {
                        console.log("WebSocket closed. Reconnecting...");
                        setTimeout(connect, 1000);
                    };
                    ws.onerror = function(error) {
                        console.error("WebSocket error:", error);
                    };
                }
            </script>
            <style>
                #messages {
                    height: 300px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                    padding: 10px;
                }
            </style>
        </body>
    </html>
    """)

@app.get("/active-streams")
async def get_active_streams():
    streams = []
    for stream_url, transcription in active_transcriptions.items():
        streams.append(transcription.get_info())
    return JSONResponse(content={"active_streams": streams})

if __name__ == "__main__":
    import os
    import subprocess

    # Run FFmpeg installation script
    if not os.path.exists(os.path.expanduser('~/.local/bin/ffmpeg')):
        subprocess.run(['bash', 'install_ffmpeg.sh'], check=True)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)