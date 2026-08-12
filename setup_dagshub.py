"""
DagsHub / MLflow authentication setup.
Run this once before training to configure your credentials.
Credentials are stored locally in .env (never committed to git).
"""
import os
from getpass import getpass
from dotenv import load_dotenv, set_key
import mlflow

ENV_FILE = ".env"

def setup_dagshub():
    load_dotenv(ENV_FILE, override=True)

    username = os.getenv("MLFLOW_TRACKING_USERNAME") or input("DagsHub username: ")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD") or getpass("DagsHub access token: ")
    project  = os.getenv("MLFLOW_TRACKING_PROJECTNAME") or input("DagsHub project name: ")

    set_key(ENV_FILE, "MLFLOW_TRACKING_USERNAME", username)
    set_key(ENV_FILE, "MLFLOW_TRACKING_PASSWORD", password)
    set_key(ENV_FILE, "MLFLOW_TRACKING_PROJECTNAME", project)

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    os.environ["MLFLOW_TRACKING_PROJECTNAME"] = project

    tracking_uri = f"https://dagshub.com/{username}/{project}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")
    return tracking_uri

if __name__ == "__main__":
    setup_dagshub()
