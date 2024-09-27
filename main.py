import logging
from fastapi import FastAPI, WebSocket, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import asyncio
import whisper
import numpy as np
import pyaudio
import wave
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
import secrets

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    api_key = Column(String, unique=True, index=True)

Base.metadata.create_all(bind=engine)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_api_key():
    return secrets.token_urlsafe(32)

async def get_api_key(api_key: str = Depends(api_key_header), db: Session = Depends(get_db)):
    if api_key is None:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key

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
    api_key = websocket.query_params.get("api_key")
    
    if not stream_url:
        await websocket.close(code=1000, reason="No stream URL provided")
        return
    
    if not api_key:
        await websocket.close(code=1000, reason="No API key provided")
        return
    
    db = next(get_db())
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        await websocket.close(code=1000, reason="Invalid API key")
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
async def home(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(request: Request, username: str = Form(...), email: str = Form(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already registered"})
    
    db_user = db.query(User).filter(User.email == email).first()
    if db_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"})
    
    new_user = User(username=username, email=email, api_key=generate_api_key())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return templates.TemplateResponse("api_key.html", {"request": request, "api_key": new_user.api_key})

@app.get("/transcribe")
async def get_transcribe_page(request: Request):
    stream_url = request.query_params.get("stream", "")
    api_key = request.query_params.get("api_key", "")
    return templates.TemplateResponse("transcribe.html", {"request": request, "stream_url": stream_url, "api_key": api_key})

@app.get("/protected")
async def protected_route(api_key: str = Depends(get_api_key)):
    return {"message": "This is a protected route"}

@app.get("/users/me")
async def read_users_me(api_key: str = Depends(get_api_key), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.api_key == api_key).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": user.username, "email": user.email}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)