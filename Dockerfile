FROM python:3.11-slim

# Install system dependencies needed for OpenCV, PIL, and other ML libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
# We explicitly allow pip to use the extra index for CPU packages (configured in requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code owned by non-root user
COPY --chown=user:user . /app

USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 8000

# Start the FastAPI application which serves the identical HTML/CSS/JS UI
# plus all /api/* inference endpoints — same UI as http://127.0.0.1:8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
