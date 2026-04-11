# Skore Webinar Demo

## Link to the live session

The recorded session is available at the following address:

https://app.livestorm.co/probabl/webinar-evaluate-compare-and-track-your-experiments-by-the-scikit-learn-founders/live?s=0044e7e9-094b-4c3a-8af9-fd5972f6efbe

## Installation

Install [pixi](https://pixi.sh) by running:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Then install the project dependencies:

```bash
pixi install
```

## Start the MLflow server

```bash
pixi run mlflow-server
```

The server will be available at http://127.0.0.1:5000.

## Start JupyterLab

In a separate terminal, run:

```bash
pixi run jupyter lab
```

Open the URL printed in the terminal and execute the demo notebook.
