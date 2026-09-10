'''
Minimal test runner for environments without pytest.

Usage (from the repository root):
    python code/tests/run_tests.py

With pytest installed, the same tests can be run with:
    PYTHONPATH=$(pwd)/code pytest code/tests -v
'''
import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_MODULES = ['test_gs_utils', 'test_view_fusion', 'test_model']


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    n_passed, n_failed = 0, 0
    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        for attr in sorted(dir(mod)):
            if not attr.startswith('test_'):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            try:
                fn()
                print(f'PASSED  {mod_name}::{attr}')
                n_passed += 1
            except Exception:
                print(f'FAILED  {mod_name}::{attr}')
                traceback.print_exc()
                n_failed += 1
    print(f'\n{n_passed} passed, {n_failed} failed')
    return 1 if n_failed else 0


if __name__ == '__main__':
    sys.exit(main())
