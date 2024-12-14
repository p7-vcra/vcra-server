#!/bin/bash

# Check if a Redis container is already running
if [ "$(docker ps -q -f name=redis)" ]; then
    echo "Redis container is already running."
else
    # Check if a Redis container exists but is stopped
    if [ "$(docker ps -aq -f name=redis)" ]; then
        echo "Starting existing Redis container..."
        docker start redis
    else
        # Run a new Redis container
        echo "Running a new Redis container..."
        docker run -d --name redis -p 6379:6379 redis
    fi
fi

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Python virtual environment not found. Creating a new one..."
    python -m venv venv
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."
source venv/bin/activate

# Install dependencies using pyproject.toml
if [ -f "pyproject.toml" ]; then
    echo "Installing Python dependencies from pyproject.toml..."
    pip install pip setuptools wheel  # Ensure the environment is ready
    pip install .
fi

echo "Starting server..."
python src/main.py