#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker could not be found. Please install Docker first."
    exit 1
fi

# Build and start the container
echo "Building and starting Docker container..."
docker compose up -d --build

# Enter the container
echo "Entering the container..."
docker compose exec lpm-dreamer bash
