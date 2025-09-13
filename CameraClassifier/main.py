import sys

# Guard against environments where Tkinter is not available (e.g., macOS Python without Tk)
try:
    import tkinter as _tk  # noqa: F401
except Exception as e:
    msg = (
        "Tkinter is not available in this Python environment.\n"
        "To run the CameraClassifier GUI, install Python with Tk support.\n\n"
        "macOS (Apple Silicon) options:\n"
        "  - If using Homebrew Python 3.13: try `brew install python-tk@3.13` (or `python-tk`)\n"
        "    and ensure your virtualenv uses the same Python.\n"
        "  - Alternatively, install the official Python from python.org (includes Tcl/Tk),\n"
        "    recreate your venv, and re-run the app.\n"
        "  - Conda users: `conda install tk`.\n\n"
        f"Original error: {e}"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)

from app import App

if __name__ == "__main__":
    App()
