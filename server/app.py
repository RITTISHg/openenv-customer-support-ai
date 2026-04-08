"""
server/app.py — Required by multi-mode deployment validator.

This module re-exports the FastAPI application from the root server.py
so that both the root-level and server/ deployment modes work correctly.
"""

import sys
import os

# Ensure root project is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app, main  # noqa: F401

__all__ = ["app", "main"]
