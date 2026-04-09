# Skore Webinar Demo

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
