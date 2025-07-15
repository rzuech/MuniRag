"""
Simple plugin architecture for MuniRAG
"""
import os
import importlib
import inspect
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PluginBase:
    """Base class for all plugins"""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = "Base plugin"
        self.enabled = True
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass
        
    def shutdown(self) -> None:
        """Cleanup when plugin is disabled"""
        pass
        
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled
        }


class DocumentProcessorPlugin(PluginBase):
    """Base class for document processing plugins"""
    
    def can_process(self, file_path: str) -> bool:
        """Check if plugin can process this file type"""
        raise NotImplementedError
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process document and return chunks"""
        raise NotImplementedError
        
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        raise NotImplementedError


class EmbeddingPlugin(PluginBase):
    """Base class for custom embedding plugins"""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        raise NotImplementedError
        
    def encode(self, texts: List[str], **kwargs) -> Any:
        """Encode texts to embeddings"""
        raise NotImplementedError
        
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        raise NotImplementedError


class StoragePlugin(PluginBase):
    """Base class for storage backend plugins"""
    
    def store(self, data: Dict[str, Any]) -> str:
        """Store data and return ID"""
        raise NotImplementedError
        
    def retrieve(self, id: str) -> Dict[str, Any]:
        """Retrieve data by ID"""
        raise NotImplementedError
        
    def search(self, query: Any, top_k: int) -> List[Dict[str, Any]]:
        """Search for similar items"""
        raise NotImplementedError


class PluginManager:
    """Manages plugin loading and lifecycle"""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, PluginBase] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        
        # Create plugin directory if it doesn't exist
        self.plugin_dir.mkdir(exist_ok=True)
        
        # Initialize hook categories
        self.hook_categories = [
            "pre_process_document",
            "post_process_document",
            "pre_embed",
            "post_embed",
            "pre_query",
            "post_query",
            "pre_store",
            "post_store"
        ]
        
        for category in self.hook_categories:
            self.hooks[category] = []
            
    def discover_plugins(self) -> Dict[str, type]:
        """Discover all available plugins"""
        discovered = {}
        
        # Check plugin directory
        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return discovered
            
        # Look for Python files in plugin directory
        for file_path in self.plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
                
            module_name = file_path.stem
            
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{module_name}", 
                    file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, PluginBase) and 
                        obj != PluginBase and
                        not name.startswith("_")):
                        discovered[name] = obj
                        logger.info(f"Discovered plugin: {name}")
                        
            except Exception as e:
                logger.error(f"Error loading plugin from {file_path}: {e}")
                
        return discovered
        
    def load_plugin(self, plugin_name: str, config: Optional[Dict] = None) -> bool:
        """Load and initialize a plugin"""
        discovered = self.discover_plugins()
        
        if plugin_name not in discovered:
            logger.error(f"Plugin {plugin_name} not found")
            return False
            
        try:
            # Instantiate plugin
            plugin_class = discovered[plugin_name]
            plugin = plugin_class()
            
            # Initialize with config
            if config:
                plugin.initialize(config)
                
            # Store plugin
            self.plugins[plugin_name] = plugin
            logger.info(f"Loaded plugin: {plugin_name}")
            
            # Register any hooks
            self._register_plugin_hooks(plugin)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
            
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            return False
            
        try:
            plugin = self.plugins[plugin_name]
            
            # Call shutdown
            plugin.shutdown()
            
            # Remove from hooks
            self._unregister_plugin_hooks(plugin)
            
            # Remove from plugins
            del self.plugins[plugin_name]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
            
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get a loaded plugin"""
        return self.plugins.get(plugin_name)
        
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins"""
        return {
            name: plugin.get_info() 
            for name, plugin in self.plugins.items()
        }
        
    def register_hook(self, category: str, callback: Callable) -> bool:
        """Register a hook callback"""
        if category not in self.hook_categories:
            logger.error(f"Unknown hook category: {category}")
            return False
            
        self.hooks[category].append(callback)
        return True
        
    def execute_hooks(self, category: str, *args, **kwargs) -> List[Any]:
        """Execute all hooks for a category"""
        if category not in self.hook_categories:
            return []
            
        results = []
        for callback in self.hooks[category]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing hook {callback.__name__}: {e}")
                
        return results
        
    def _register_plugin_hooks(self, plugin: PluginBase) -> None:
        """Register hooks from a plugin"""
        # Check for hook methods
        for category in self.hook_categories:
            method_name = f"hook_{category}"
            if hasattr(plugin, method_name):
                method = getattr(plugin, method_name)
                if callable(method):
                    self.register_hook(category, method)
                    
    def _unregister_plugin_hooks(self, plugin: PluginBase) -> None:
        """Unregister hooks from a plugin"""
        # Remove plugin's hooks
        for category in self.hook_categories:
            method_name = f"hook_{category}"
            if hasattr(plugin, method_name):
                method = getattr(plugin, method_name)
                if method in self.hooks[category]:
                    self.hooks[category].remove(method)
                    
    def get_processors_for_file(self, file_path: str) -> List[DocumentProcessorPlugin]:
        """Get all processors that can handle a file"""
        processors = []
        
        for plugin in self.plugins.values():
            if isinstance(plugin, DocumentProcessorPlugin):
                if plugin.can_process(file_path):
                    processors.append(plugin)
                    
        return processors


# Example plugins
class WordDocumentPlugin(DocumentProcessorPlugin):
    """Plugin for processing Word documents"""
    
    def __init__(self):
        super().__init__()
        self.name = "WordDocumentProcessor"
        self.description = "Processes .docx files"
        
    def get_supported_extensions(self) -> List[str]:
        return [".docx", ".doc"]
        
    def can_process(self, file_path: str) -> bool:
        return any(file_path.endswith(ext) for ext in self.get_supported_extensions())
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Word document"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Simple chunking
            chunks = []
            words = text.split()
            for i in range(0, len(words), 500):
                chunk_text = " ".join(words[i:i+500])
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": file_path,
                        "type": "docx"
                    }
                })
                
            return chunks
            
        except ImportError:
            logger.error("python-docx not installed")
            return []
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            return []


class MarkdownDocumentPlugin(DocumentProcessorPlugin):
    """Plugin for processing Markdown documents"""
    
    def __init__(self):
        super().__init__()
        self.name = "MarkdownProcessor"
        self.description = "Processes .md files"
        
    def get_supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]
        
    def can_process(self, file_path: str) -> bool:
        return any(file_path.endswith(ext) for ext in self.get_supported_extensions())
        
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Markdown document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by headers
            import re
            sections = re.split(r'\n#{1,3}\s+', content)
            
            chunks = []
            for i, section in enumerate(sections):
                if section.strip():
                    chunks.append({
                        "content": section.strip(),
                        "metadata": {
                            "source": file_path,
                            "type": "markdown",
                            "section": i
                        }
                    })
                    
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing Markdown: {e}")
            return []


class CustomAnalyticsPlugin(PluginBase):
    """Example analytics plugin with hooks"""
    
    def __init__(self):
        super().__init__()
        self.name = "CustomAnalytics"
        self.description = "Tracks usage analytics"
        self.query_count = 0
        self.document_count = 0
        
    def hook_pre_query(self, query: str, **kwargs):
        """Hook called before query processing"""
        self.query_count += 1
        logger.info(f"Analytics: Query #{self.query_count}: {query[:50]}...")
        
    def hook_post_process_document(self, file_path: str, chunks: List[Dict], **kwargs):
        """Hook called after document processing"""
        self.document_count += 1
        logger.info(f"Analytics: Processed document #{self.document_count}: {file_path}")
        logger.info(f"Analytics: Created {len(chunks)} chunks")
        
    def get_stats(self) -> Dict[str, int]:
        """Get analytics stats"""
        return {
            "queries": self.query_count,
            "documents": self.document_count
        }


# Test function
def test_plugin_system():
    """Test the plugin system"""
    # Create plugin manager
    pm = PluginManager()
    
    # Create test plugins directory
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    # Save example plugin
    plugin_code = '''
from src.plugin_manager import DocumentProcessorPlugin

class TestPlugin(DocumentProcessorPlugin):
    def __init__(self):
        super().__init__()
        self.name = "TestPlugin"
        self.description = "Test plugin for .txt files"
        
    def get_supported_extensions(self):
        return [".txt"]
        
    def can_process(self, file_path):
        return file_path.endswith(".txt")
        
    def process(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        return [{"content": content, "metadata": {"source": file_path}}]
'''
    
    with open(plugins_dir / "test_plugin.py", "w") as f:
        f.write(plugin_code)
    
    # Test discovery
    print("Discovering plugins...")
    discovered = pm.discover_plugins()
    print(f"Found {len(discovered)} plugins")
    
    # Test loading
    if "TestPlugin" in discovered:
        print("\nLoading TestPlugin...")
        pm.load_plugin("TestPlugin")
        
        # Test usage
        processors = pm.get_processors_for_file("test.txt")
        print(f"Found {len(processors)} processors for .txt files")
        
    # List loaded plugins
    print("\nLoaded plugins:")
    for name, info in pm.list_plugins().items():
        print(f"  - {name}: {info}")


if __name__ == "__main__":
    test_plugin_system()