"""Memory client wrapper module.

Re-exports mem0 MemoryClient for convenience.
The main bot implementation is in services/bot.py.
"""
from mem0 import MemoryClient

__all__ = ["MemoryClient"]
