import sys
import os

# Make `engine` importable: adds this file's directory so `import engine` finds engine/__init__.py
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.gui import main

if __name__ == "__main__":
    main()
