# chronotopia

```bash
# Clone the repository
git clone https://github.com/borfebor/chronotopia.git

# Navigate to the folder containing the cloned repository
cd chrono_app

# Build the Chrono app (the -t flag specifies the name of the Docker image)
docker build -t chrono_app . 

# Start the Chrono app from the terminal
docker run -p 8501:8501 chrono-app
``