"""Frozen-app entry point.

PyInstaller needs a real script, not a console_script entry point. Keep this
tiny: everything of substance lives in the package so it stays testable.
"""

import multiprocessing
import sys


def main() -> int:
    from openforge.cli import main as cli_main

    return int(cli_main() or 0)


if __name__ == "__main__":
    multiprocessing.freeze_support()  # no-op unless frozen on Windows
    sys.exit(main())
