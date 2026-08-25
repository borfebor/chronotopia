# Chronotopia

**Version 0.8.0** · [Documentation](https://borfebor.github.io/chronotopia/) ·
[Release notes](RELEASE_NOTES.md) · [How to cite](CITATION.cff)

Analysis of time-course data for circadian biology: preprocessing, period
estimation, rhythmicity testing, ~108 features per sample, plate-format analysis
and publication-ready figures.

## Running it


1. Ensure [Docker](www.docker.com/get-started) is Installed and Running
Make sure you have Docker installed on your system and that the Docker daemon is running.

2. Clone and Build the Application
```bash
# Clone the repository
git clone https://github.com/borfebor/chronotopia.git

# Navigate to the folder containing the cloned repository
cd chronotopia

# Build the Chrono app (the -t flag specifies the name of the Docker image)
docker build -t chronotopia . 

# Start the Chrono app from the terminal
docker run -p 8501:8501 chronotopia
``` 

3. Access the Web Interface:
Once the container is running, open the following link in your browser:
[http://localhost:8501/](http://localhost:8501/)

## Licence

Chronotopia is free software under the
[GNU General Public License v3 or later](LICENSE).

This is required rather than chosen: Chronotopia uses
[pyBOAT](https://github.com/tensionhead/pyBOAT) for its wavelet analysis, and
pyBOAT is GPL-3.0-or-later, so the combined work must be distributed under
compatible terms.

Copyright (C) 2026 Borja Ferrero-Bordera, Ludwig-Maximilians-Universität
München, and the John Innes Centre.
