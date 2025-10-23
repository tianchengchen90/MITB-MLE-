# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Set the working directory *early*
# This directory will be owned by root by default
WORKDIR /app

# Copy the requirements file into the container
# This will also be owned by root
COPY requirements.txt ./

# Switch to root to install packages and fix permissions
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        procps \
        bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # === NEW ===
    # Give the airflow user ownership of the /app directory and its contents
    chown -R airflow:airflow /app

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Switch back to the airflow user
USER airflow

# Install Python dependencies using requirements.txt
# This now runs as the 'airflow' user in a directory they own
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks (this part is fine)
# Note: This just declares that /app *can* be a volume.
# It doesn't actively mount anything during the build.
VOLUME /app