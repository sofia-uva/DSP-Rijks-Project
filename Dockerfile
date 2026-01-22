# --- Base image ---
FROM python:3.11-slim

# --- Working directory ---
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# --- Copy requirements and install Python packages ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy app files ---
COPY . .

# --- Set HF_TOKEN environment variable (will use Space secret) ---
ENV HF_TOKEN=$HF_TOKEN

# --- Run Streamlit ---
CMD ["streamlit", "run", "app2.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
