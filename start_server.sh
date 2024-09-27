#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update system
sudo apt update
sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3 python3-pip python3-venv ffmpeg nginx supervisor certbot python3-certbot-nginx

# Set up the application directory
APP_DIR="/opt/fastapi_app"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Create and activate virtual environment
python3 -m venv $APP_DIR/venv
source $APP_DIR/venv/bin/activate

# Install Python dependencies
pip install fastapi uvicorn yt-dlp whisper numpy pyaudio psutil

# Copy your application code
# Replace with your actual method of getting the code (e.g., git clone)
echo "Please copy your application code to $APP_DIR"
read -p "Press enter when you've copied your code"

# Set up Supervisor
sudo tee /etc/supervisor/conf.d/fastapi_app.conf > /dev/null <<EOT
[program:fastapi_app]
command=$APP_DIR/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
directory=$APP_DIR
user=$USER
autostart=true
autorestart=true
stderr_logfile=/var/log/fastapi_app.err.log
stdout_logfile=/var/log/fastapi_app.out.log
EOT

sudo supervisorctl reread
sudo supervisorctl update

# Set up Nginx
sudo tee /etc/nginx/sites-available/fastapi_app > /dev/null <<EOT
server {
    listen 80;
    server_name \$host;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOT

sudo ln -s /etc/nginx/sites-available/fastapi_app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Set up SSL
read -p "Enter your domain name: " DOMAIN_NAME
sudo certbot --nginx -d $DOMAIN_NAME

# Set up firewall
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

# Start the application
sudo supervisorctl start fastapi_app

echo "Setup complete! Your FastAPI application should now be running at https://$DOMAIN_NAME"
echo "Please make sure to update your application to use wss:// for WebSocket connections."