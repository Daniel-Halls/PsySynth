FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY psysynth/ psysynth/
COPY data/ data/
COPY assets/ assets/

# Install the application and its dependencies
RUN pip install --no-cache-dir .

# Expose the Dash server port
EXPOSE 8050

# Run the Dash app
CMD ["psysynth-dashboard"]
