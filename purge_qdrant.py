#!/usr/bin/env python3
"""
Manual Qdrant purge script
Use this to manually purge collections when needed
"""

import sys
from src.qdrant_manager import get_qdrant_manager
from src.logger import get_logger

logger = get_logger("purge_script")


def main():
    """Main function to purge Qdrant collections"""
    
    # Confirm with user
    print("\n⚠️  WARNING: This will delete all Qdrant collections!")
    print("Are you sure you want to continue? (yes/no): ", end="")
    
    response = input().strip().lower()
    
    if response != "yes":
        print("Purge cancelled.")
        return
    
    # Get manager and purge
    manager = get_qdrant_manager()
    
    print("\nPurging all collections...")
    deleted = manager.purge_all_collections()
    
    print(f"\n✅ Purge complete. Deleted {deleted} collections.")


if __name__ == "__main__":
    main()