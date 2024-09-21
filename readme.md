Start the virtual environment
```
python -m venv venv
source bin/activate
```

Install the dependencies
```
pip install -r requirements.txt
```

Install ffmpeg
```
brew install ffmpeg
export PATH="/opt/homebrew/bin:$PATH"
```

Install portaudio
```
brew install portaudio
```

Run the server
```
uvicorn main:app --reload
```

How to use
```
http://localhost:8000/?stream=<stream_url>
```

Notes:
- Only tested locally on Mac M3
- The virtual audio cable is "BlackHole 2ch"
- Only tested with YouTube streams


Todo:
- [ ] Make it so you can convert the stream to text directly instead of saving to a file
- [ ] Explore other models for transcription
- [ ] Deploy to a cloud service
- [ ] Perform latency testing


Disclaimer:
- I do not condone or encourage copyright infringement. This is for educational purposes only.
- I am not responsible for any misuse of this software.
- Use it at your own risk.
