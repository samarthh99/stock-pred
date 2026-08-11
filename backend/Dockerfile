FROM python:3.10-slim

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . .

# Pull LFS files (they are copied as pointers, need to pull)
RUN git lfs install && git lfs pull

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
