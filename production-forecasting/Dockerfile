FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (PyTorch CPU only from official index, fall back to PyPI)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

# Copy application code
COPY endpoint_server.py forecasting.py smoothing.py visualize_helper.py ./

# Expose port
EXPOSE 5000

# Generate self-signed certificates on startup and run server
CMD ["sh", "-c", "openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost' 2>/dev/null || true && python endpoint_server.py"]
