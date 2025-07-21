# MuniRAG Documentation Index

Welcome to the MuniRAG documentation. This index helps you navigate our comprehensive documentation.

## 🚀 Getting Started
- [README.md](../README.md) - Project overview and quick start
- [NEWCOMER_GUIDE.md](NEWCOMER_GUIDE.md) - Introduction for new developers

## 📋 Project Management
- [MASTER_PRIORITY_LIST.md](MASTER_PRIORITY_LIST.md) - Prioritized task list
- [CHANGELOG_PERFORMANCE.md](CHANGELOG_PERFORMANCE.md) - Performance improvements log

## 🏗️ Architecture & Design
- [EMBEDDING_ARCHITECTURE.md](EMBEDDING_ARCHITECTURE.md) - Embedding system design
- [MULTI_MODEL_ARCHITECTURE.md](MULTI_MODEL_ARCHITECTURE.md) - Multi-model support
- [LLM_ARCHITECTURE.md](LLM_ARCHITECTURE.md) - Language model integration
- [GPU_RESOURCE_MANAGEMENT.md](GPU_RESOURCE_MANAGEMENT.md) - GPU optimization

## 🧪 Testing
- **[testing/AUTOMATED_ACCURACY_TESTING.md](testing/AUTOMATED_ACCURACY_TESTING.md)** ⭐ - Complete testing guide
- [TESTING.md](TESTING.md) - General testing guidelines
- [RUN_GPU_TESTS.md](RUN_GPU_TESTS.md) - GPU performance testing

## 🔧 Operations
- [OPERATIONS.md](OPERATIONS.md) - Operational procedures
- [QDRANT_DATA_MANAGEMENT.md](QDRANT_DATA_MANAGEMENT.md) - Database management
- [SCRIPTS_DOCUMENTATION.md](SCRIPTS_DOCUMENTATION.md) - Utility scripts reference
- [REBUILD_INSTRUCTIONS.md](REBUILD_INSTRUCTIONS.md) - Rebuild procedures

## 🚀 Deployment
- [deployment/](deployment/) - Deployment configurations
- [UPGRADE_V2.md](UPGRADE_V2.md) - Upgrade guide

## 📚 Features & Enhancements
- [SEMANTIC_CHUNKING_ALGORITHMS.md](SEMANTIC_CHUNKING_ALGORITHMS.md) - Text chunking
- [OCR_IMAGE_HANDLING.md](OCR_IMAGE_HANDLING.md) - OCR support
- [ENSEMBLE_ARCHITECTURE_RESEARCH.md](ENSEMBLE_ARCHITECTURE_RESEARCH.md) - Multi-model research
- [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) - Roadmap

## 🗄️ Archived Documentation
- [archived/](archived/) - Historical documentation

## Quick Links by Topic

### For Testing & Quality Assurance
1. Start with [testing/AUTOMATED_ACCURACY_TESTING.md](testing/AUTOMATED_ACCURACY_TESTING.md)
2. Review test questions in `test_questions.json`
3. Run tests with `./quick_test_fixed.sh`

### For System Configuration
1. Check [QDRANT_DATA_MANAGEMENT.md](QDRANT_DATA_MANAGEMENT.md)
2. Review `.env.example` for all settings
3. See [GPU_RESOURCE_MANAGEMENT.md](GPU_RESOURCE_MANAGEMENT.md) for optimization

### For Development
1. Read [NEWCOMER_GUIDE.md](NEWCOMER_GUIDE.md)
2. Check [MASTER_PRIORITY_LIST.md](MASTER_PRIORITY_LIST.md) for tasks
3. Follow [OPERATIONS.md](OPERATIONS.md) for procedures

## Documentation Standards

### File Organization
- **Main docs**: Core documentation in `/docs`
- **Testing docs**: Testing-specific in `/docs/testing`
- **Deployment**: Production configs in `/docs/deployment`
- **Archived**: Historical docs in `/docs/archived`

### Naming Conventions
- UPPERCASE.md for major documents
- lowercase directories
- Descriptive names (not abbreviations)

### Content Guidelines
- Start with overview/purpose
- Include practical examples
- Add troubleshooting sections
- Keep updated with code changes