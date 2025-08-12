FROM ubuntu:latest

# Install Python and venv
RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /venv

# Activate venv and install packages
RUN /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir torch pandas mlflow numpy
# Set the default shell to use the venv
ENV PATH="/venv/bin:$PATH"