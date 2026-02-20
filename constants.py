"""Path constants for OpenDeepResearch environment."""
from pathlib import Path
import os

# Check production vs local environment
# Production: /orwd_data exists (OpenReward cloud storage)
# Local: Use current directory for development
if os.path.exists("/orwd_data"):
    ENV_PATH = Path("/orwd_data")
else:
    ENV_PATH = Path(__file__).parent

# Dataset file paths
TRAIN_JSONL = ENV_PATH / "train.jsonl"
TEST_JSONL = ENV_PATH / "test.jsonl"
