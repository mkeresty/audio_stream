#!/bin/bash
mkdir -p ~/.local/bin
curl -L https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz | tar Jxf - -C ~/.local/bin --strip-components=1 ffmpeg-master-latest-linux64-gpl/bin/ffmpeg
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc