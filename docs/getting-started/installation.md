# Installation

Chronotopia is a Streamlit application. Docker is the supported route because it
pins the whole stack, including R and MetaCycle; a local install is fine if you
are happy to manage that yourself.

## Docker

=== "First run"

    ```bash
    git clone https://github.com/borfebor/chronotopia.git
    cd chronotopia
    docker build -t chronotopia .
    docker run -p 8501:8501 chronotopia
    ```

=== "Afterwards"

    ```bash
    docker run -p 8501:8501 chronotopia
    ```

Then open <http://localhost:8501>.

The build takes a few minutes the first time — it installs R and the MetaCycle
package alongside the Python stack. Later runs start in seconds.

!!! tip "Keeping your files"

    The container has no access to your filesystem, which is usually what you
    want. Upload through the browser and download results the same way. If you
    would rather mount a directory:

    ```bash
    docker run -p 8501:8501 -v "$PWD/data:/data" chronotopia
    ```

## Local Python

Python 3.11 is what the Docker image uses and what CI tests against.

```bash
git clone https://github.com/borfebor/chronotopia.git
cd chronotopia
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Two dependencies worth knowing about

**`scikit-learn` is pinned to 1.8.0.** The bundled random-forest classifier was
trained under that version and the version string is baked into the pickle.
Unpickling under a different one warns first and eventually fails outright. If
you need to move it, retrain the model rather than unpinning.

**`rpy2` needs a working R.** It is what MetaCycle runs through, so the
`meta2d`, `JTK`, `ARS` and `LS` testing methods need R installed with the
MetaCycle package:

```r
install.packages("MetaCycle")
```

Without R, everything else still works. `PermCosinor` and `Tempo` are pure
Python, and all five period-estimation methods are unaffected — you simply lose
the MetaCycle entries from the **Testing method** menu.

## Checking the install

Open the app. In the sidebar, switch on **Generate example dataset**. If a
lineplot appears, the numeric stack is working.

To check the analysis paths more thoroughly, run the two harnesses:

```bash
python verify.py                        # 332 checks over the whole app
python tutorials/verify_tutorial_data.py  # 40 checks over the tutorial datasets
```

## Building the documentation

This site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

```bash
pip install -r requirements-docs.txt
python docs/make_figures.py     # renders figures, stages the datasets
mkdocs serve
```

`make_figures.py` imports `methods.py` and `styles.py`, so it needs the app's
numeric dependencies — but not `rpy2`. It must be run before `mkdocs serve`,
because it is what puts the downloadable datasets under `docs/downloads/`.
