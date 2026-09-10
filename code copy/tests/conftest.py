import os
import sys

# make `models` / `datasets` importable, matching scripts/env.sh
# (export PYTHONPATH=$(pwd)/code:$PYTHONPATH)
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
