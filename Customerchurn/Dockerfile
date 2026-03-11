FROM python:3.11-slim

# Prevents Python from writing .pyc files & ensures logs flush to console
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but good for scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Flask runs on 5000
EXPOSE 5000

# Production server
CMD ["gunicorn", "-b", "0.0.0.0:5000", "application:application"]