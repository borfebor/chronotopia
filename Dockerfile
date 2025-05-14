FROM python:3.11-slim

# Install R and dependencies
RUN apt-get update && apt-get install -y \
    r-base \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages (MetaCycle needs BiocManager and dependencies)
RUN R -e "install.packages('BiocManager', repos='http://cran.us.r-project.org')" \
 && R -e "BiocManager::install('MetaCycle', ask = FALSE, update = FALSE)"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app
WORKDIR /app

# Set Streamlit to run
CMD ["streamlit", "run", "app.py"]
