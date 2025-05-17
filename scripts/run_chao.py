#!/usr/bin/env python3
"""
Run Chao Assistant
Simple script to start the Chao assistant chatbot
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.chao_assistant import interactive_chat

if __name__ == "__main__":
    print("Starting Chao AI Personal Assistant...")
    interactive_chat() 