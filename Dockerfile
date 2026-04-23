# Use a lightweight official Python runtime as the base image.
FROM python:3.11-slim

# Set /app as the working directory inside the container.
WORKDIR /app

# Copy dependency file first to leverage Docker layer caching.
COPY requirements.txt .
# Install Python dependencies required by the application.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project source code into the container.
COPY . .

# Document that the app listens on port 5000.
EXPOSE 5000

# Start the Flask app when the container launches.
CMD ["python", "app.py"]
