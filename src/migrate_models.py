#!/usr/bin/env python3
"""
Command-line tool for migrating embeddings between different models.

Usage:
    python migrate_models.py --source "BAAI/bge-large-en-v1.5" --target "thenlper/gte-large"
    python migrate_models.py --list-collections
    python migrate_models.py --check-compatibility "model1" "model2"
"""

import argparse
import sys
from typing import List, Optional
from datetime import datetime
import json

from src.vector_store_v2 import MultiModelVectorStore
from src.embedder import EmbeddingModel
from src.config_v2 import enhanced_settings
from src.model_migration import ModelMigrationManager
from src.logger import get_logger

logger = get_logger("migrate_models_cli")


class ModelMigrationCLI:
    """Command-line interface for model migration"""
    
    def __init__(self):
        self.vector_store = MultiModelVectorStore()
        self.migration_manager = ModelMigrationManager()
    
    def list_collections(self) -> None:
        """List all collections with their information"""
        print("\n📊 Current Collections:")
        print("-" * 80)
        
        collections = self.vector_store.get_collection_stats()
        
        if not collections:
            print("No collections found.")
            return
        
        # Table header
        print(f"{'Model':<40} {'Documents':<15} {'Dimensions':<12} {'Collection':<30}")
        print("-" * 80)
        
        total_docs = 0
        for collection_name, info in collections.items():
            model_name = info.get('model_name', 'Unknown')
            doc_count = info['points_count']
            dimension = info['dimension']
            
            total_docs += doc_count
            
            print(f"{model_name:<40} {doc_count:<15,} {dimension:<12} {collection_name:<30}")
        
        print("-" * 80)
        print(f"Total documents across all collections: {total_docs:,}")
        print()
    
    def check_compatibility(self, model1: str, model2: str) -> None:
        """Check if two models are compatible"""
        print(f"\n🔍 Checking compatibility between models:")
        print(f"  Model 1: {model1}")
        print(f"  Model 2: {model2}")
        print("-" * 60)
        
        try:
            dim1 = enhanced_settings.get_embedding_dimension(model1)
            dim2 = enhanced_settings.get_embedding_dimension(model2)
            
            print(f"  {model1}: {dim1} dimensions")
            print(f"  {model2}: {dim2} dimensions")
            
            if dim1 == dim2:
                print("\n✅ Models are compatible (same dimensions)")
                print("   You can switch between these models without re-embedding.")
            else:
                print("\n❌ Models are NOT compatible (different dimensions)")
                print("   Migration will require re-embedding all documents.")
            
        except Exception as e:
            print(f"\n❌ Error checking compatibility: {str(e)}")
    
    def migrate(
        self, 
        source_model: str, 
        target_model: str,
        delete_source: bool = False,
        batch_size: int = 100,
        dry_run: bool = False
    ) -> None:
        """Perform migration between models"""
        print(f"\n🚀 Migration Plan:")
        print(f"  Source: {source_model}")
        print(f"  Target: {target_model}")
        print(f"  Delete source after migration: {delete_source}")
        print(f"  Batch size: {batch_size}")
        print("-" * 60)
        
        # Check if source collection exists
        source_collection = self.vector_store.collection_manager.get_collection_name(source_model)
        collections = self.vector_store.get_collection_stats()
        
        if source_collection not in collections:
            print(f"\n❌ Source collection not found: {source_collection}")
            return
        
        source_info = collections[source_collection]
        doc_count = source_info['points_count']
        
        if doc_count == 0:
            print(f"\n⚠️  Source collection is empty. Nothing to migrate.")
            return
        
        print(f"\n📊 Found {doc_count:,} documents to migrate")
        
        # Check compatibility
        source_dim = enhanced_settings.get_embedding_dimension(source_model)
        target_dim = enhanced_settings.get_embedding_dimension(target_model)
        
        if source_dim != target_dim:
            print(f"\n⚠️  Dimension mismatch detected:")
            print(f"   Source: {source_dim}D")
            print(f"   Target: {target_dim}D")
            print("   Documents will be re-embedded with the target model.")
        
        # Estimate time
        estimated_time = self.migration_manager.estimate_migration_time(
            source_model, target_model, doc_count
        )
        print(f"\n⏱️  Estimated time: {int(estimated_time//60)}m {int(estimated_time%60)}s")
        
        if dry_run:
            print("\n🔍 DRY RUN - No changes will be made.")
            return
        
        # Confirm
        print("\nProceed with migration? (yes/no): ", end="")
        confirmation = input().strip().lower()
        
        if confirmation != "yes":
            print("Migration cancelled.")
            return
        
        # Initialize embedder
        print(f"\n🔧 Initializing {target_model}...")
        try:
            embedder = EmbeddingModel(target_model)
        except Exception as e:
            print(f"❌ Failed to initialize target model: {str(e)}")
            return
        
        # Define embedding function
        def embed_func(texts: List[str]) -> List[List[float]]:
            return embedder.embed_documents(texts)
        
        # Progress callback
        def progress_callback(progress: float, message: str):
            bar_length = 50
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\r[{bar}] {progress*100:.1f}% - {message}", end="", flush=True)
        
        # Start migration
        print("\n\n🚀 Starting migration...")
        migration_start = datetime.now()
        
        try:
            result = self.vector_store.migrate_collection(
                source_model=source_model,
                target_model=target_model,
                embedder_func=embed_func,
                batch_size=batch_size,
                progress_callback=progress_callback
            )
            
            print()  # New line after progress bar
            
            migration_end = datetime.now()
            duration = (migration_end - migration_start).total_seconds()
            
            if result['status'] == 'success':
                print(f"\n✅ Migration completed successfully!")
                print(f"   Documents migrated: {result['migrated']:,}")
                print(f"   Time taken: {int(duration//60)}m {int(duration%60)}s")
                print(f"   Speed: {result['migrated']/duration:.1f} docs/sec")
                
                # Delete source if requested
                if delete_source:
                    print(f"\n🗑️  Deleting source collection...")
                    if self.vector_store.delete_collection(source_model, confirm=True):
                        print("   Source collection deleted successfully.")
                    else:
                        print("   ⚠️  Failed to delete source collection.")
                
                # Save migration record
                self.migration_manager.add_migration_record({
                    "timestamp": migration_start.isoformat(),
                    "source_model": source_model,
                    "target_model": target_model,
                    "documents_migrated": result['migrated'],
                    "duration_seconds": duration,
                    "status": result['status'],
                    "deleted_source": delete_source
                })
                
            else:
                print(f"\n❌ Migration failed: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Migration interrupted by user.")
        except Exception as e:
            print(f"\n\n❌ Migration error: {str(e)}")
            logger.error(f"Migration error: {str(e)}", exc_info=True)
    
    def show_history(self, limit: int = 10) -> None:
        """Show migration history"""
        print("\n📜 Migration History:")
        print("-" * 80)
        
        history = self.migration_manager.get_migration_history()
        
        if not history:
            print("No migrations performed yet.")
            return
        
        # Show most recent first
        for record in reversed(history[-limit:]):
            timestamp = datetime.fromisoformat(record['timestamp'])
            duration = record['duration_seconds']
            
            print(f"\n{timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  {record['source_model']} → {record['target_model']}")
            print(f"  Documents: {record['documents_migrated']:,}")
            print(f"  Duration: {int(duration//60)}m {int(duration%60)}s")
            print(f"  Status: {record['status']}")
            
            if record.get('deleted_source'):
                print("  Note: Source collection was deleted")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Model Migration Tool for MuniRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all collections
  python migrate_models.py --list-collections
  
  # Check model compatibility
  python migrate_models.py --check-compatibility "BAAI/bge-large-en-v1.5" "thenlper/gte-large"
  
  # Migrate between models
  python migrate_models.py --source "BAAI/bge-large-en-v1.5" --target "thenlper/gte-large"
  
  # Migrate and delete source
  python migrate_models.py --source "old-model" --target "new-model" --delete-source
  
  # Dry run (no changes)
  python migrate_models.py --source "model1" --target "model2" --dry-run
  
  # Show migration history
  python migrate_models.py --history
        """
    )
    
    # Actions
    parser.add_argument(
        "--list-collections", "-l",
        action="store_true",
        help="List all collections with their information"
    )
    
    parser.add_argument(
        "--check-compatibility", "-c",
        nargs=2,
        metavar=("MODEL1", "MODEL2"),
        help="Check if two models are compatible"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Source model to migrate from"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=str,
        help="Target model to migrate to"
    )
    
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source collection after successful migration"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration (default: 100)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration plan without executing"
    )
    
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show migration history"
    )
    
    parser.add_argument(
        "--history-limit",
        type=int,
        default=10,
        help="Number of history entries to show (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = ModelMigrationCLI()
    
    # Execute requested action
    if args.list_collections:
        cli.list_collections()
    
    elif args.check_compatibility:
        cli.check_compatibility(args.check_compatibility[0], args.check_compatibility[1])
    
    elif args.source and args.target:
        cli.migrate(
            source_model=args.source,
            target_model=args.target,
            delete_source=args.delete_source,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    
    elif args.history:
        cli.show_history(limit=args.history_limit)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()