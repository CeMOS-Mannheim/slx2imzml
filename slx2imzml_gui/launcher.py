import pathlib
import subprocess
import sys


def main() -> int:
    app_path = pathlib.Path(__file__).with_name("app.py")
    command = [sys.executable, "-m", "shiny", "run", str(app_path)]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())