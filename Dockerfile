# Step 1 — Start from official Python image
FROM python:3.11-slim

# Step 2 — Set working directory inside container
WORKDIR /app

# Step 3 — Copy requirements first (for caching)
COPY requirements.txt .

# Step 4 — Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5 — Copy entire project into container
COPY . .

# Step 6 — Create logs directory inside container
RUN mkdir -p logs

# Step 7 — Expose port 8000
EXPOSE 8000

# Step 8 — Command to start the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]