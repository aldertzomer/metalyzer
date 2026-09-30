#!/usr/bin/env python3
"""Compatibility entry point for the modular local Ministral classifier.

The implementation lives in modules/llm.py and is also available through
``python metalyzer.py --method llm``.
"""
from __future__ import annotations

import sys

import metalyzer


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--method" in arguments:
        index = arguments.index("--method")
        if index + 1 >= len(arguments) or arguments[index + 1] != "llm":
            raise SystemExit("metalyzer_local_mistral.py requires --method llm")
    else:
        arguments.extend(("--method", "llm"))
    metalyzer.main(arguments)


if __name__ == "__main__":
    main()
