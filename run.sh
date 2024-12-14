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

echo "Starting server..."
python src/main.py