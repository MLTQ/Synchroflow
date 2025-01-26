FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install system packages needed for compiling Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your Python script and required files into the container
COPY testmodel3.py . 
COPY transformer_model.pth . 
COPY labeled_data.csv . 

# Install Python libraries using pip, including tqdm and joblib
RUN pip3 install --no-cache-dir torch pandas numpy scikit-learn tqdm joblib

# (Optional) Make the script executable if it has a shebang line
RUN chmod +x testmodel3.py

# Define the default command to run your script
CMD ["python3", "testmodel3.py"]

