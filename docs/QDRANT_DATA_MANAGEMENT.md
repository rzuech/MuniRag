# Qdrant Data Management Guide

## Overview
MuniRAG includes automated data management features for Qdrant to support both development and production environments.

## Configuration Settings

### Primary Setting: `RESET_DATA_ON_STARTUP`
- **Purpose**: Controls whether Qdrant collections are purged when the application starts
- **Default**: `false` (production-safe, preserves data)
- **Location**: `src/config.py` line 111

#### Values:
- `true` - Purges collections on every restart (DEVELOPMENT MODE)
- `false` - Preserves all data across restarts (PRODUCTION MODE)

### Secondary Setting: `RESET_MUNIRAG_ONLY`
- **Purpose**: Safety feature that limits which collections get purged
- **Default**: `true`
- **Location**: `src/config.py` line 112
- **Note**: Only applies when `RESET_DATA_ON_STARTUP=true`

#### Values:
- `true` - Only purges collections starting with "munirag_" (SAFER)
- `false` - Purges ALL collections in Qdrant (DANGEROUS!)

## How to Configure

### Method 1: Environment Variables (Recommended)

Add to your `.env` file:

```bash
# For PRODUCTION (preserves data)
RESET_DATA_ON_STARTUP=false
RESET_MUNIRAG_ONLY=true

# For DEVELOPMENT (purges on restart)
RESET_DATA_ON_STARTUP=true
RESET_MUNIRAG_ONLY=true
```

### Method 2: Docker Compose Override

In `docker-compose.yml`:

```yaml
services:
  munirag:
    environment:
      - RESET_DATA_ON_STARTUP=false  # Production mode
      - RESET_MUNIRAG_ONLY=true      # Safety net
```

### Method 3: Command Line Override

```bash
# One-time development run with purge
docker-compose run -e RESET_DATA_ON_STARTUP=true munirag

# Production run (preserve data)
docker-compose run -e RESET_DATA_ON_STARTUP=false munirag
```

## Behavior Matrix

| RESET_DATA_ON_STARTUP | RESET_MUNIRAG_ONLY | Result |
|----------------------|-------------------|---------|
| false | true/false | No purging - all data preserved |
| true | true | Only munirag_* collections purged |
| true | false | ALL collections purged (dangerous!) |

## Best Practices

1. **For Production Deployments**:
   - Always set `RESET_DATA_ON_STARTUP=false`
   - Keep `RESET_MUNIRAG_ONLY=true` as a safety net
   - Use `.env` file for consistency

2. **For Development**:
   - Set `RESET_DATA_ON_STARTUP=true` for clean starts
   - Keep `RESET_MUNIRAG_ONLY=true` to protect other data
   - Consider using command-line overrides for flexibility

3. **Safety First**:
   - Never set `RESET_MUNIRAG_ONLY=false` unless you're absolutely sure
   - Always backup important data before changing these settings
   - Test configuration changes in development first

## Manual Data Management

### Purge All Collections Manually

```bash
# Interactive script with confirmation
python purge_qdrant.py
```

### Check Current Collections

```bash
docker exec munirag-munirag-1 python -c "
from src.qdrant_manager import get_qdrant_manager
manager = get_qdrant_manager()
manager.health_check()
"
```

## Troubleshooting

### Data Not Being Preserved
- Check that `RESET_DATA_ON_STARTUP=false` in your `.env` file
- Verify with: `docker-compose config | grep RESET`

### Wrong Collections Being Purged
- Ensure `RESET_MUNIRAG_ONLY=true`
- Check collection names - only "munirag_*" should be affected

### Need Fresh Start in Development
- Temporarily set `RESET_DATA_ON_STARTUP=true`
- Or use manual purge: `python purge_qdrant.py`