import logging
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import asyncio
import whisper
import numpy as np
import pyaudio
import wave
import threading
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import warnings
import yt_dlp
import subprocess
import io
import os
import psutil
from collections import deque
import datetime

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

model = whisper.load_model("base.en")  # Load the English-only base model

# Audio recording settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

# Flags to control behavior
USE_YT_DLP = True
SAVE_TO_FILE = False

BUFFER_SIZE = 10  # Number of audio chunks to buffer
CHUNK_DURATION = RECORD_SECONDS  # Duration of each chunk in seconds

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

async def get_live_audio_url(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = await asyncio.to_thread(ydl.extract_info, youtube_url, download=False)
        return info['url']

async def record_audio_yt_dlp(youtube_url):
    audio_url = await get_live_audio_url(youtube_url)
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
    
    buffer = deque(maxlen=BUFFER_SIZE)
    start_time = time.time()
    
    while True:
        audio_chunk = await process.stdout.read(CHUNK * CHANNELS * 2 * RECORD_SECONDS)
        if not audio_chunk:
            break
        
        current_time = time.time()
        chunk_age = current_time - start_time
        buffer.append((audio_chunk, chunk_age, current_time))
        
        if len(buffer) == BUFFER_SIZE:
            oldest_chunk, oldest_age, oldest_timestamp = buffer[0]
            if oldest_age > BUFFER_SIZE * CHUNK_DURATION:
                # We've buffered enough old data, start yielding from the most recent
                for chunk, _, timestamp in reversed(buffer):
                    yield chunk, timestamp
                buffer.clear()
            else:
                yield oldest_chunk, oldest_timestamp
                buffer.popleft()
        
        start_time = current_time
    
    process.terminate()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    stream_url = websocket.query_params.get("stream")
    if not stream_url:
        await websocket.close(code=1000, reason="No stream URL provided")
        return

    if USE_YT_DLP:
        audio_generator = record_audio_yt_dlp(stream_url)
        
        try:
            async for audio_chunk, chunk_timestamp in audio_generator:
                chunk_start_time = time.time()
                
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                
                transcription_start = time.time()
                result = model.transcribe(audio_np, language="en")
                transcription_time = time.time() - transcription_start
                
                latency = time.time() - chunk_start_time
                performance_monitor.update_metrics(transcription_time, latency)
                
                if result["text"]:
                    timestamp = datetime.datetime.fromtimestamp(chunk_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    await websocket.send_json({
                        "type": "transcription",
                        "text": f"[{timestamp}] {result['text']}"
                    })
                    logger.info(f"Transcribed: [{timestamp}] {result['text']}")
                
                await websocket.send_json({
                    "type": "performance",
                    **performance_monitor.get_metrics()
                })
                
        except Exception as e:
            logger.error(f"Error in websocket_endpoint (yt-dlp): {str(e)}")
        finally:
            await websocket.close()
    else:
        # Selenium logic remains unchanged
        ...

@app.get("/")
async def get(request: Request):
    stream_url = request.query_params.get("stream", "")
    return HTMLResponse(f"""
        <html>
            <body>
                <h1>WebSocket Transcription</h1>
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
                    function connect() {{
                        var stream_url = "{stream_url}";
                        ws = new WebSocket(`ws://localhost:8000/ws?stream=${{encodeURIComponent(stream_url)}}`);
                        ws.onmessage = function(event) {{
                            var data = JSON.parse(event.data);
                            if (data.type === "transcription") {{
                                var messages = document.getElementById('messages');
                                var message = document.createElement('li');
                                var content = document.createTextNode(data.text);
                                message.appendChild(content);
                                messages.appendChild(message);
                                messages.scrollTop = messages.scrollHeight;
                            }} else if (data.type === "performance") {{
                                document.getElementById('avg_time').textContent = data.avg_transcription_time.toFixed(3);
                                document.getElementById('avg_latency').textContent = data.avg_latency.toFixed(3);
                                document.getElementById('memory').textContent = data.memory_usage_mb.toFixed(2);
                                document.getElementById('cpu').textContent = data.cpu_usage_percent.toFixed(2);
                                document.getElementById('uptime').textContent = data.uptime_seconds.toFixed(0);
                                document.getElementById('count').textContent = data.transcription_count;
                            }}
                        }};
                        ws.onclose = function(event) {{
                            console.log("WebSocket closed. Reconnecting...");
                            setTimeout(connect, 1000);
                        }};
                        ws.onerror = function(error) {{
                            console.error("WebSocket error:", error);
                        }};
                    }}
                    connect();
                </script>
                <style>
                    #messages {{
                        height: 300px;
                        overflow-y: auto;
                        border: 1px solid #ccc;
                        padding: 10px;
                    }}
                </style>
            </body>
        </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
