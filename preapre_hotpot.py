"""
DEPRECATED shim — kept for backward compatibility.

Use `python -m prepare_hotpot` instead.
This file will be removed in a future release.
"""

import warnings

warnings.warn(
    "preapre_hotpot is deprecated (typo). Use `python -m prepare_hotpot` instead. "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from prepare_hotpot import main  # noqa: F401

if __name__ == "__main__":
    main()

# python -m preapre_hotpot  (deprecated alias)
# prefer: python -m prepare_hotpot
