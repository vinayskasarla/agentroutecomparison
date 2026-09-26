"""Load settings from a .env file next to the app (KEY=value lines). Variables already set in the environment win,
so the same code runs locally from .env and in AWS from Secrets Manager. No dependency needed."""
import os

PATH = os.environ.get("ENV_FILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load(path=PATH):
    if not os.path.exists(path):
        return []
    loaded = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


LOADED = load()
