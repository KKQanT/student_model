#!/bin/bash

# Check if venv exists and create it if necessary
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
fi

# Check if data folder exists and create it if necessary
if [ ! -d "data" ]; then
    echo "Creating data folder..."
    mkdir data
else
    echo "data folder already exists."
fi

# Check if results folder exists and create it if necessary
if [ ! -d "results" ]; then
    echo "Creating results folder..."
    mkdir results
else
    echo "results folder already exists."
fi
