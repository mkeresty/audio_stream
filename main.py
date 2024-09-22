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

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.transcription_count = 0
        self.total_transcription_time = 0
        self.start_time = time.time()
        self.process = psutil.Process()

    def update_transcription_time(self, transcription_time):
        self.transcription_count += 1
        self.total_transcription_time += transcription_time

    def get_average_transcription_time(self):
        if self.transcription_count == 0:
            return 0
        return self.total_transcription_time / self.transcription_count

    def get_memory_usage(self):
        return self.process.memory_info().rss / 1024 / 1024  # in MB

    def get_cpu_usage(self):
        return self.process.cpu_percent()

    def get_uptime(self):
        return time.time() - self.start_time

performance_monitor = PerformanceMonitor()

def get_virtual_cable_index():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        logger.info(f"Audio device {i}: {dev['name']}")
        if "BlackHole" in dev['name']:
            return i
    raise Exception("Virtual audio cable not found")

def record_audio_selenium(stop_event):
    try:
        p = pyaudio.PyAudio()
        virtual_cable_index = get_virtual_cable_index()
        logger.info(f"Using virtual cable index: {virtual_cable_index}")
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=virtual_cable_index,
                        frames_per_buffer=CHUNK)

        frames = []

        while not stop_event.is_set():
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    except Exception as e:
        logger.error(f"Error in record_audio: {str(e)}")

def get_audio_url(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def record_audio_yt_dlp(youtube_url, stop_event):
    audio_url = get_audio_url(youtube_url)
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', audio_url,
        '-f', 's16le',
        '-ar', str(RATE),
        '-ac', str(CHANNELS),
        '-'
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    while not stop_event.is_set():
        audio_chunk = process.stdout.read(CHUNK * CHANNELS * 2)
        if not audio_chunk:
            break
        yield audio_chunk
    
    process.terminate()

def save_audio_to_file(audio_data):
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes per sample for s16le
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    stream_url = websocket.query_params.get("stream")
    if not stream_url:
        await websocket.send_text("Error: No stream URL provided")
        return

    if USE_YT_DLP:
        stop_event = threading.Event()
        audio_generator = record_audio_yt_dlp(stream_url, stop_event)
        
        try:
            while True:
                audio_chunks = b''.join([next(audio_generator) for _ in range(int(RATE * RECORD_SECONDS / CHUNK))])
                
                transcription_start = time.time()
                if SAVE_TO_FILE:
                    save_audio_to_file(audio_chunks)
                    result = model.transcribe(WAVE_OUTPUT_FILENAME, language="en")
                else:
                    audio_np = np.frombuffer(audio_chunks, dtype=np.int16).astype(np.float32) / 32768.0
                    result = model.transcribe(audio_np, language="en")
                transcription_time = time.time() - transcription_start
                performance_monitor.update_transcription_time(transcription_time)
                
                if result["text"]:
                    await websocket.send_text(result["text"])
                    logger.info(f"Transcribed: {result['text']}")
                
                # Send performance metrics
                performance_data = {
                    "avg_transcription_time": performance_monitor.get_average_transcription_time(),
                    "memory_usage_mb": performance_monitor.get_memory_usage(),
                    "cpu_usage_percent": performance_monitor.get_cpu_usage(),
                    "uptime_seconds": performance_monitor.get_uptime(),
                    "transcription_count": performance_monitor.transcription_count
                }
                await websocket.send_json(performance_data)
                
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in websocket_endpoint (yt-dlp): {str(e)}")
            await websocket.send_text(f"Error: {str(e)}")
        finally:
            stop_event.set()
            if SAVE_TO_FILE and os.path.exists(WAVE_OUTPUT_FILENAME):
                os.remove(WAVE_OUTPUT_FILENAME)
    else:
        chrome_options = Options()
        chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
        driver = None

        try:
            driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome WebDriver initialized successfully")

            driver.get(stream_url)
            logger.info(f"Navigated to {stream_url}")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "video.html5-main-video"))
            )
            logger.info("Video player found")

            driver.execute_script("document.querySelector('video').play()")
            logger.info("Video playback started")

            time.sleep(2)

            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio_selenium, args=(stop_event,))
            recording_thread.start()

            while True:
                await asyncio.sleep(RECORD_SECONDS)
                stop_event.set()
                recording_thread.join()

                result = model.transcribe(WAVE_OUTPUT_FILENAME, language="en")
                
                if result["text"]:
                    await websocket.send_text(result["text"])
                    logger.info(f"Transcribed: {result['text']}")

                stop_event.clear()
                recording_thread = threading.Thread(target=record_audio_selenium, args=(stop_event,))
                recording_thread.start()

        except Exception as e:
            logger.error(f"Error in websocket_endpoint (Selenium): {str(e)}")
            await websocket.send_text(f"Error: {str(e)}")
        finally:
            stop_event.set()
            if driver:
                driver.quit()

@app.get("/")
async def get(request: Request):
    stream_url = request.query_params.get("stream", "")
    return HTMLResponse(f"""
        <html>
            <body>
                <h1>WebSocket Transcription</h1>
                <ul id='messages'>
                </ul>
                <div id='performance'>
                    <h2>Performance Metrics</h2>
                    <p>Average Transcription Time: <span id='avg_time'></span> seconds</p>
                    <p>Memory Usage: <span id='memory'></span> MB</p>
                    <p>CPU Usage: <span id='cpu'></span>%</p>
                    <p>Uptime: <span id='uptime'></span> seconds</p>
                    <p>Transcription Count: <span id='count'></span></p>
                </div>
                <script>
                    var ws = new WebSocket("ws://localhost:8000/ws?stream={stream_url}");
                    ws.onmessage = function(event) {{
                        var data = JSON.parse(event.data);
                        if (data.text) {{
                            var messages = document.getElementById('messages')
                            var message = document.createElement('li')
                            var content = document.createTextNode(data.text)
                            message.appendChild(content)
                            messages.appendChild(message)
                        }} else {{
                            document.getElementById('avg_time').textContent = data.avg_transcription_time.toFixed(3);
                            document.getElementById('memory').textContent = data.memory_usage_mb.toFixed(2);
                            document.getElementById('cpu').textContent = data.cpu_usage_percent.toFixed(2);
                            document.getElementById('uptime').textContent = data.uptime_seconds.toFixed(0);
                            document.getElementById('count').textContent = data.transcription_count;
                        }}
                    }};
                </script>
            </body>
        </html>
    """)

def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"Index {i}: {dev['name']}")

if __name__ == "__main__":
    list_audio_devices()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
