# Data Upload Requirements for OpenDeepResearch

## Overview
This environment requires the Open-DeepResearch dataset to be uploaded to OpenReward cloud storage.

## Files to Upload

Upload to namespace: `EnvCommons/opendeepresearch`

Directory structure on /orwd_data:
```
/orwd_data/opendeepresearch/
├── train.jsonl  (2,216 examples, ~611KB)
└── test.jsonl   (100 examples, ~1.7MB with reference conversations)
```

## Download Source

Download from HuggingFace:

```bash
# Download train split
curl -L "https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch/resolve/main/train.jsonl" -o train.jsonl

# Download test split
curl -L "https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch/resolve/main/test.jsonl" -o test.jsonl
```

## File Descriptions

- **train.jsonl**: Training data with queries only (no reference conversations)
  - Format: `{"query": "...", "reference": null}`
  - 2,216 examples
  - Used for RL training and open-ended evaluation
  - Mixed Chinese and English queries

- **test.jsonl**: Evaluation data with queries and reference conversations
  - Format: `{"query": "...", "reference": [...]}`
  - 100 examples
  - Reference contains full multi-turn conversation with tool calls
  - Used for benchmarking

## Upload Instructions

The user will upload these files to the OpenReward namespace storage following platform guidelines at https://openreward.ai.

## API Keys Required

This environment requires two API keys to function:

1. **OpenAI API Key** - For LLM-based grading using gpt-5-mini with ArenaRL 7-dimension rubric
   - Get yours at: https://platform.openai.com/api-keys
   - Required permission: Model access to gpt-5-mini

2. **Tavily API Key** - For web search and URL content extraction
   - Get yours at: https://tavily.com
   - Required for: web_search and web_fetch tools

Pass both keys when creating a session:
```python
async with environment.session(
    task=task,
    secrets={
        "openai_api_key": "sk-...",
        "tavily_api_key": "tvly-..."
    }
) as session:
    ...
```
