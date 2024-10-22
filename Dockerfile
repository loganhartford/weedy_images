FROM tensorflow/tensorflow:latest-gpu-jupyter

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install additional system dependencies
RUN apt-get update && \
    apt-get install -y graphviz && \
    rm -rf /var/lib/apt/lists/*

# Install Kaggle CLI for dataset downloading
RUN pip install kaggle

# Create a directory for Kaggle API credentials
RUN mkdir -p /root/.kaggle

# Set up Jupyter to use the /tf/notebooks directory
WORKDIR /tf/notebooks

# Expose the Jupyter notebook port
EXPOSE 8888
