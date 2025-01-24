FROM python:3.9-slim

WORKDIR /app

# Install system packages needed for compiling python-alsaaudio and for ALSA itself
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libasound2 \
    libasound2-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy your Python scripts into the container
COPY sound.py .

# Install Python libraries (including python-alsaaudio)
RUN pip3 install --no-cache-dir numpy python-alsaaudio

# Make scripts executable
RUN chmod +x sound.py

# Run sound.py on container start
CMD ["python3", "sound.py"] 