# OpenDeepResearch Environment

OpenReward environment for the **Open-DeepResearch** dataset from Alibaba-NLP, featuring multi-turn research tasks with web search and ArenaRL-based evaluation.

## Overview

This environment implements the Open-DeepResearch benchmark for evaluating AI agents on complex, open-ended research tasks. Agents use web search and content fetching tools to conduct thorough investigations, then submit comprehensive reports evaluated using the ArenaRL 7-dimension rubric.

### Key Features

- **Multi-turn research**: Agents conduct iterative web searches and content analysis
- **Bilingual support**: Automatic detection and instruction provision for Chinese and English queries
- **ArenaRL grading**: 7-dimension parallel evaluation (Framework, Tool Usage, Coverage, Relevance, Accuracy, Depth, Clarity)
- **Enhanced tools**: Advanced search depth, larger content windows, optimized for comprehensive research
- **Both splits**: 2,216 training tasks + 100 test tasks

### Dataset

- **Source**: [Alibaba-NLP/Open-DeepResearch](https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch)
- **Paper**: [ArenaRL: Scaling RL for Open-Ended Agents](https://arxiv.org/abs/2601.06487)
- **Train split**: 2,216 tasks with queries only (no references)
- **Test split**: 100 tasks with queries + reference conversations
- **Languages**: Mixed Chinese and English
- **Task types**: Technical writing, ideation, explanation, open-ended research

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/EnvCommons/opendeepresearch.git
cd opendeepresearch

# Download datasets
curl -L "https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch/resolve/main/train.jsonl" -o train.jsonl
curl -L "https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch/resolve/main/test.jsonl" -o test.jsonl

# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."

# Run server
python server.py
```

### Docker

```bash
docker build -t opendeepresearch:latest .
docker run -p 8080:8080 opendeepresearch:latest
```

## Usage

### Agent Workflow

1. **Receive query**: Agent gets a research question (Chinese or English)
2. **Web search**: Use `web_search(query)` tool to find relevant information
   - Returns titles, URLs, and snippets
   - Can search multiple times from different angles
3. **Fetch content**: Use `fetch_url(url)` to read full page content
   - Extract detailed information from promising sources
4. **Submit report**: Use `submit_report(...)` with:
   - `report`: Comprehensive research report (500-10,000 chars)
   - `key_findings`: List of 3-10 key findings
   - `sources_cited`: List of source URLs (min 1)
   - `confidence`: Confidence level (0.0-1.0)
5. **Receive grading**: Get reward (0.0-1.0) with 7-dimension scores

### Example

```python
import asyncio
from openai import AsyncOpenAI
from openreward import AsyncOpenReward

async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()

    environment = or_client.environments.get(name="EnvCommons/opendeepresearch")
    tasks = await environment.list_tasks(split="train")
    tools = await environment.list_tools(format="openai")

    async with environment.session(
        task=tasks[0],
        secrets={
            "openai_api_key": "sk-...",
            "tavily_api_key": "tvly-..."
        }
    ) as session:
        prompt = await session.get_prompt()
        # Use agent to conduct research and submit report
        ...

if __name__ == "__main__":
    asyncio.run(main())
```

See `test_agent.py` for a complete working example.

## Tools

### `web_search(query: str)`

Search the web using Tavily API.

- **Search depth**: "advanced" (more comprehensive than basic)
- **Max results**: 8
- **Returns**: Titles, URLs, snippets

Example:
```python
result = await session.call_tool("web_search", {"query": "AI safety research 2025"})
```

### `fetch_url(url: str)`

Fetch full content from a URL using Tavily Extract API.

- **Max length**: 12,000 characters
- **Returns**: Full page text content

Example:
```python
result = await session.call_tool("fetch_url", {"url": "https://arxiv.org/abs/2601.06487"})
```

### `submit_report(...)`

Submit final research report for grading.

**Parameters**:
- `report` (str): Research report (500-10,000 chars)
- `key_findings` (List[str]): 3-10 key findings
- `sources_cited` (List[str]): Source URLs (min 1)
- `confidence` (float): Confidence (0.0-1.0)

**Returns**:
- Reward: 0.0-1.0 (weighted average of 7 dimensions)
- Dimension scores: Framework, Tool Usage, Coverage, Relevance, Accuracy, Depth, Clarity
- Feedback: Overall assessment

Example:
```python
result = await session.call_tool("submit_report", {
    "report": "...",
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "sources_cited": ["https://...", "https://..."],
    "confidence": 0.8
})
```

## ArenaRL 7-Dimension Rubric

Reports are evaluated on 7 dimensions in parallel using gpt-5-mini:

1. **Framework** (15%): Structural completeness and logical coherence
2. **Tool Usage** (15%): Appropriateness and efficiency of tool invocations
3. **Coverage** (15%): Sufficiency of retrieved information
4. **Relevance** (15%): How well responses address user queries
5. **Accuracy** (15%): Factual correctness and consistency
6. **Depth** (15%): Analytical depth and reasoning coherence
7. **Clarity** (10%): Organization, readability, and practical usability

Each dimension is scored 0.0-1.0, then combined with weights to produce final reward.

**Reference**: [ArenaRL Paper](https://arxiv.org/abs/2601.06487) - Section on Open-DeepResearch evaluation

## API Keys

This environment requires two API keys:

1. **OpenAI API Key** - For LLM grading with gpt-5-mini
   - Get at: https://platform.openai.com/api-keys

2. **Tavily API Key** - For web search and content extraction
   - Get at: https://tavily.com

Pass both when creating a session:
```python
secrets={
    "openai_api_key": "sk-...",
    "tavily_api_key": "tvly-..."
}
```

## File Structure

```
opendeepresearch/
├── opendeepresearch.py    # Main environment class
├── server.py              # Minimal server wrapper
├── test_agent.py          # Agent testing script
├── constants.py           # Path handling
├── requirements.txt       # Dependencies
├── Dockerfile            # Container configuration
├── DATA_UPLOAD.md        # Data upload instructions
├── README.md             # This file
└── .gitignore
```

## Data Upload

For deployment on OpenReward cloud, datasets must be uploaded to `/orwd_data/opendeepresearch/`.

See [DATA_UPLOAD.md](DATA_UPLOAD.md) for detailed instructions.

## Testing

### Syntax Check
```bash
python -m py_compile *.py
```

### Local Server
```bash
python server.py
# Server starts at http://0.0.0.0:8080
```

### Agent Integration
```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
python test_agent.py
```

## Citation

If you use this environment, please cite the ArenaRL paper:

```bibtex
@misc{zhang2026arenarlscalingrlopenended,
  title={ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking},
  author={Qiang Zhang and others},
  year={2026},
  eprint={2601.06487},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.06487}
}
```

## License

This environment implementation is provided for research and educational purposes. The underlying dataset (Open-DeepResearch) is licensed under CC BY-NC 4.0 by Alibaba-NLP.

## Links

- **Dataset**: https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch
- **Paper**: https://arxiv.org/abs/2601.06487
- **OpenReward**: https://openreward.ai
- **GitHub**: https://github.com/EnvCommons/opendeepresearch
