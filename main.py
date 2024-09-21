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

def get_virtual_cable_index():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        logger.info(f"Audio device {i}: {dev['name']}")
        if "BlackHole" in dev['name']:
            return i
    raise Exception("Virtual audio cable not found")

def record_audio(stop_event):
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Get the stream URL from the query parameters
    stream_url = websocket.query_params.get("stream")
    if not stream_url:
        await websocket.send_text("Error: No stream URL provided")
        return

    chrome_options = Options()
    chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
    driver = None

    try:
        driver = webdriver.Chrome(options=chrome_options)
        logger.info("Chrome WebDriver initialized successfully")

        driver.get(stream_url)
        logger.info(f"Navigated to {stream_url}")

        # Wait for the video player to be ready
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "video.html5-main-video"))
        )
        logger.info("Video player found")

        # Start playing the video
        driver.execute_script("document.querySelector('video').play()")
        logger.info("Video playback started")

        # Wait a bit for the audio to start
        time.sleep(2)

        stop_event = threading.Event()
        recording_thread = threading.Thread(target=record_audio, args=(stop_event,))
        recording_thread.start()

        while True:
            # Record for a few seconds
            await asyncio.sleep(RECORD_SECONDS)
            stop_event.set()
            recording_thread.join()

            # Transcribe the recorded audio, forcing English language
            result = model.transcribe(WAVE_OUTPUT_FILENAME, language="en")
            
            if result["text"]:
                await websocket.send_text(result["text"])
                logger.info(f"Transcribed: {result['text']}")

            # Reset for next recording
            stop_event.clear()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event,))
            recording_thread.start()

    except Exception as e:
        logger.error(f"Error in websocket_endpoint: {str(e)}")
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
                <script>
                    var ws = new WebSocket("ws://localhost:8000/ws?stream={stream_url}");
                    ws.onmessage = function(event) {{
                        var messages = document.getElementById('messages')
                        var message = document.createElement('li')
                        var content = document.createTextNode(event.data)
                        message.appendChild(content)
                        messages.appendChild(message)
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
