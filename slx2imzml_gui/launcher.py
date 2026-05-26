"""Entry point — launch the Flask GUI and open a browser tab."""
import webbrowser
import threading
from slx2imzml_gui.app import app


def main():
    port = 5001
    url = f"http://localhost:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"SCiLS Exporter running at {url}")
    app.run(host="localhost", port=port, debug=False, use_reloader=False, threaded=False)


if __name__ == "__main__":
    main()
