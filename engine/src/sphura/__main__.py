"""Allows `python -m sphura`, which keeps working if the folder is moved."""

import sys

from sphura.cli import main

sys.exit(main())
