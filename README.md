# Open-DeepResearch

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/opendeepresearch) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch)

## Description

Open-DeepResearch is an environment for evaluating agents on autonomous information retrieval and research report generation. Introduced alongside the ArenaRL framework by Alibaba-NLP, it tasks agents with conducting web-based research on complex queries and producing structured reports with cited sources and key findings.

## Capabilities

- Autonomous web-based research and information gathering
- Generating structured research reports with citations
- Multi-hop information retrieval and synthesis
- Bilingual support (Chinese and English)

## Compute Requirements

This is a multi-turn environment with no sandbox. Agents interact through web search and URL fetching tools only.

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Tasks

There are two splits in this environment:

- **Train**: 2,216 research queries
- **Test**: 100 research queries

Each task presents a research question. The agent must conduct web searches, gather information, and submit a comprehensive research report.

## Reward Structure

This is a multi-turn environment with continuous reward (0.0–1.0):

Scoring uses the ArenaRL 7-dimension rubric, with each dimension graded in parallel by gpt-5-mini:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Framework | 15% | Structural organization of the report |
| Tool Usage | 15% | Effective use of search and retrieval tools |
| Coverage | 15% | Breadth of topics and perspectives covered |
| Relevance | 15% | Alignment with the research question |
| Accuracy | 15% | Factual correctness of claims |
| Depth | 15% | Analytical depth and insight |
| Clarity | 10% | Writing quality and readability |

The final reward is the weighted average across all 7 dimensions. Submissions must include at least 3 key findings and 1 cited source.

## Data

Data consists of two JSONL files (`train.jsonl` with 2,216 queries, `test.jsonl` with 100 queries). Each instance contains a research query; test instances also include reference conversations. Bilingual support covers both Chinese and English queries.

Source: [Alibaba-NLP/Open-DeepResearch](https://huggingface.co/datasets/Alibaba-NLP/Open-DeepResearch)

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using Tavily API (advanced depth, up to 8 results). |
| `fetch_url` | Fetch content from a specific URL (max 12,000 characters). |
| `submit_report` | Submit a research report with key findings, cited sources, and confidence score. |

## Time Horizon

Open-DeepResearch is a multi-turn environment. Agents perform multiple rounds of web search and content fetching before compiling and submitting a research report.

## Environment Difficulty

The ArenaRL paper evaluates models on Open-DeepResearch using pairwise win rates:

| Model | Mean Win Rate | Valid Generation |
|-------|---------------|------------------|
| ArenaRL (fine-tuned) | 64.3% | 99.0% |
| Grok-4 | 34.8% | 83.0% |
| Gemini-2.5-pro | 28.3% | 92.0% |
| Claude-3.7-Sonnet | 19.1% | 89.0% |
| GPT-4o | 12.2% | 88.0% |

Traditional RL methods struggle with long-horizon research tasks due to "length bias", often producing unusable outputs (17–32% valid generation rate).

## Other Environment Requirements

- **OpenAI API key**: Required for 7-dimension rubric grading via gpt-5-mini
- **Tavily API key**: Required for web search and URL content extraction

Pass via `secrets={"openai_api_key": "...", "tavily_api_key": "..."}`.

## Safety

Agents in Open-DeepResearch conduct web searches and generate research reports. The environment involves real web access, so agents may encounter unfiltered web content. Reports should be treated as AI-generated and not used as authoritative sources without verification.

## Citations

```bibtex
@article{zhang2026arenarl,
  author    = {Qiang Zhang and Boli Chen and Fanrui Zhang and Ruixue Ding and Shihang Wang and Qiuchen Wang and Yinfeng Huang and Haonan Zhang and Rongxiang Zhu and Pengyong Wang and Ailin Ren and Xin Li and Pengjun Xie and Jiawei Liu and Ning Guo and Jingren Zhou and Zheng-Jun Zha},
  title     = {ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking},
  journal   = {arXiv preprint arXiv:2601.06487},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.06487}
}
```
