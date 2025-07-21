#!/usr/bin/env python3
"""Fix Qdrant by purging all data and starting fresh"""

from src.qdrant_manager import get_qdrant_manager

print("Fixing Qdrant...")
manager = get_qdrant_manager()

# Purge all collections
manager.purge_all_collections()

# Verify it's clean
manager.health_check()

print("\nQdrant is now clean and ready for use!")
print("Please re-upload your PDFs through the web interface.")