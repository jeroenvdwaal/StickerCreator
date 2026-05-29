#!/usr/bin/env python3
"""Sticker Creator - AI-powered sticker extraction for KDE.

Usage:
    python3 main.py              # Run the application
    sticker-creator              # After 'pip install .'
"""

import sys
import os

# Ensure the project root is in the path for development runs
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sticker_creator.app import StickerCreatorApp


def main():
    """Application entry point."""
    app = StickerCreatorApp(sys.argv)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
