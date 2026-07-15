"""Pytest configuration — adds project root to sys.path so all ARIA imports work."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
