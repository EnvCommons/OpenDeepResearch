"""
OpenDeepResearch Environment - Multi-turn research with ArenaRL grading

A multi-turn research environment where agents use web search and content fetching
to conduct open-ended research, then submit comprehensive reports evaluated using
the ArenaRL 7-dimension rubric.

Dataset: Alibaba-NLP/Open-DeepResearch (HuggingFace)
- Train: 2,216 tasks (query only, no references)
- Test: 100 tasks (query + reference conversations)

Paper: https://arxiv.org/abs/2601.06487
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List

import openai
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, Server, TextBlock, ToolOutput, tool

from constants import TRAIN_JSONL, TEST_JSONL


# Pydantic schemas for type safety
class TaskSpec(BaseModel):
    """Task specification for OpenDeepResearch environment"""
    id: str
    query: str
    split: str
    has_reference: bool


class WebSearchInput(BaseModel):
    """Parameters for web_search tool"""
    query: str = Field(..., description="Search query string")


class FetchUrlInput(BaseModel):
    """Parameters for fetch_url tool"""
    url: str = Field(..., description="URL to fetch content from")


class SubmitReportInput(BaseModel):
    """Parameters for submit_report tool"""
    report: str = Field(
        ...,
        description="Comprehensive research report (minimum 500 characters, maximum 10,000)"
    )
    key_findings: List[str] = Field(
        ...,
        description="List of 3-10 key findings from research",
        min_length=3,
        max_length=10
    )
    sources_cited: List[str] = Field(
        ...,
        description="List of URLs used as sources (minimum 1)",
        min_length=1
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in report quality (0.0 to 1.0)"
    )


# Bilingual instruction templates
CHINESE_INSTRUCTIONS = """您的任务是对这个研究问题进行深入调查。

可用工具:
1. web_search(query: str) - 搜索网络信息
   - 返回标题、URL和摘要
   - 可以多次搜索以从不同角度收集信息
2. fetch_url(url: str) - 获取特定URL的完整内容
   - 在web_search之后使用此工具阅读完整页面
   - 有助于提取详细信息
3. submit_report(...) - 提交最终研究报告（结束任务）

说明:
1. 使用web_search查找相关信息
   - 考虑搜索问题中提到的关键实体、日期或概念
   - 问题通常需要跨多个搜索的多跳推理
2. 使用fetch_url获取有价值URL的完整内容
3. 准备好后，使用submit_report提交:
   - report: 全面的研究报告（至少500字符，最多10,000字符）
   - key_findings: 3-10个关键发现列表
   - sources_cited: 使用的URL列表（至少1个）
   - confidence: 您的信心水平（0.0到1.0）

重要提示: 这个问题需要深入研究。请充分调查后再提交报告。"""


ENGLISH_INSTRUCTIONS = """Your task is to conduct in-depth research on this question.

Available Tools:
1. web_search(query: str) - Search for information
   - Returns titles, URLs, and snippets
   - You can search multiple times from different angles
2. fetch_url(url: str) - Fetch full content from a specific URL
   - Use after web_search to read complete pages
   - Helpful for extracting detailed information
3. submit_report(...) - Submit your final research report (ends task)

Instructions:
1. Use web_search to find relevant information
   - Consider searching for key entities, dates, or concepts mentioned
   - Questions often require multi-hop reasoning across multiple searches
2. Use fetch_url to get complete content from promising URLs
3. When ready, use submit_report with:
   - report: Comprehensive research report (minimum 500 characters, maximum 10,000)
   - key_findings: List of 3-10 key findings
   - sources_cited: List of URLs used as sources (minimum 1)
   - confidence: Your confidence level (0.0 to 1.0)

Important: This question requires thorough research. Investigate thoroughly before submitting."""


def get_task_queries(split: str) -> List[Dict]:
    """
    Load only queries for task listing (lightweight).
    Does NOT load reference conversations to minimize memory usage.

    Args:
        split: "train" or "test"

    Returns:
        List of task specification dicts with id, query, split, has_reference

    Raises:
        FileNotFoundError: If dataset file not found
        ValueError: If split is invalid
    """
    if split not in ["train", "test"]:
        raise ValueError(f"Unknown split: {split}. Available: train, test")

    path = TRAIN_JSONL if split == "train" else TEST_JSONL

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"See DATA_UPLOAD.md for download instructions."
        )

    tasks = []
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            tasks.append({
                "id": f"opendeepresearch_{split}_{idx}",
                "query": data["query"],
                "split": split,
                "has_reference": data.get("reference") is not None
            })

    return tasks


class OpenDeepResearch(Environment):
    """
    OpenDeepResearch environment: Multi-turn research with ArenaRL 7-dimension grading.

    Agent workflow:
    1. Receives a research question (Chinese or English)
    2. Uses web_search tool to gather information (multiple searches allowed)
    3. Uses fetch_url tool to read detailed content from URLs
    4. Submits comprehensive research report via submit_report
    5. Report is graded by gpt-5-mini on 7 ArenaRL dimensions in parallel
    6. Receives reward (0.0-1.0) with dimension scores and feedback

    ArenaRL Rubric (7 dimensions):
    - Framework: Structural completeness and logical coherence
    - Tool Usage: Appropriateness and efficiency of tool invocations
    - Coverage: Sufficiency of retrieved information
    - Relevance: How well responses address user queries
    - Accuracy: Factual correctness and consistency
    - Depth: Analytical depth and reasoning coherence
    - Clarity: Organization, readability, and practical usability

    Reference: https://arxiv.org/abs/2601.06487
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize OpenDeepResearch environment.

        Args:
            task_spec: Task specification with id, query, split, has_reference
            secrets: Must contain "openai_api_key" for grading and "tavily_api_key" for search

        Raises:
            ValueError: If required API keys missing or task_spec invalid
        """
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        # CRITICAL: Validate secrets upfront, NEVER fall back to env vars
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets parameter for LLM grading. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "tavily_api_key required in secrets parameter for web search. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        # Initialize clients - MUST use secrets, NEVER environment variables
        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available dataset splits."""
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """
        List all tasks for a given split.

        Returns lightweight task specs (query only, no reference conversations).

        Args:
            split: "train" or "test"

        Returns:
            List of task specification dicts

        Raises:
            ValueError: If split is invalid
            FileNotFoundError: If dataset file not found
        """
        return get_task_queries(split)

    async def get_prompt(self) -> List[TextBlock]:
        """
        Generate research prompt for agent with bilingual support.

        Returns:
            List with single TextBlock containing the research question and instructions
        """
        # Detect if query is Chinese or English (simple heuristic)
        is_chinese = any('\u4e00' <= c <= '\u9fff' for c in self.config.query)

        # Provide instructions in matching language
        instructions = CHINESE_INSTRUCTIONS if is_chinese else ENGLISH_INSTRUCTIONS

        prompt_text = f"""Research Question: {self.config.query}

{instructions}"""

        return [TextBlock(type="text", text=prompt_text)]

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web using Tavily. Returns search results with titles, URLs, and snippets.
        Use fetch_url tool to get full content from specific URLs if needed.
        """
        try:
            # Use Tavily search API with advanced depth for research
            response = await self.tavily_client.search(
                query=params.query,
                search_depth="advanced",  # More thorough than BrowseComp's "basic"
                max_results=8  # More results for comprehensive research
            )

            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text="No search results found.")],
                    metadata={"query": params.query, "results": []},
                    reward=0.0,
                    finished=False
                )

            # Build display text
            display_parts = [f"Search results for: {params.query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("content", "")
                display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

            display_text = "\n".join(display_parts)

            return ToolOutput(
                blocks=[TextBlock(type="text", text=display_text)],
                metadata={
                    "query": params.query,
                    "results": results,
                    "count": len(results)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Web search failed: {str(e)}")],
                metadata={"query": params.query, "error": str(e)},
                reward=0.0,
                finished=False
            )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the full text content from a specific URL using Tavily's extract method.
        Use this after web_search to get complete information from a page.
        """
        try:
            response = await self.tavily_client.extract(urls=[params.url])

            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text=f"No content extracted from {params.url}")],
                    metadata={"url": params.url, "results": []},
                    reward=0.0,
                    finished=False
                )

            result = results[0]
            raw_content = result.get("raw_content", "")

            # Larger truncation limit for deep research (vs 8000 in BrowseComp)
            max_length = 12000
            if len(raw_content) > max_length:
                raw_content = raw_content[:max_length] + "...\n[Content truncated]"

            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Content from {params.url}:\n\n{raw_content}")],
                metadata={
                    "url": params.url,
                    "length": len(raw_content)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Failed to fetch URL: {str(e)}")],
                metadata={"url": params.url, "error": str(e)},
                reward=0.0,
                finished=False
            )

    @tool
    async def submit_report(self, params: SubmitReportInput) -> ToolOutput:
        """
        Submit your final research report.
        """

        if len(params.key_findings) < 3:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Insufficient key findings ({len(params.key_findings)}). Minimum 3 required.")],
                metadata={"error": "insufficient_findings"},
                reward=0.0,
                finished=True
            )

        if len(params.sources_cited) < 1:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="No sources cited. At least 1 source required.")],
                metadata={"error": "no_sources"},
                reward=0.0,
                finished=True
            )

        # Grade the report using LLM with ArenaRL 7-dimension rubric
        grading_result = await self._grade_report(
            report=params.report,
            key_findings=params.key_findings,
            sources_cited=params.sources_cited,
            confidence=params.confidence
        )

        reward = grading_result["reward"]
        scores = grading_result["scores"]

        # Format display output
        display_text = f"""Research Report Submitted

Quality Score: {reward:.2f}/1.0

ArenaRL Dimension Scores:
- Framework: {scores.get('framework', 0.0):.2f}
- Tool Usage: {scores.get('tool_usage', 0.0):.2f}
- Coverage: {scores.get('coverage', 0.0):.2f}
- Relevance: {scores.get('relevance', 0.0):.2f}
- Accuracy: {scores.get('accuracy', 0.0):.2f}
- Depth: {scores.get('depth', 0.0):.2f}
- Clarity: {scores.get('clarity', 0.0):.2f}

Feedback: {grading_result['feedback']}

Report Length: {len(params.report)} characters
Key Findings: {len(params.key_findings)}
Sources Cited: {len(params.sources_cited)}
Confidence: {params.confidence:.2f}
"""

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "task_id": self.config.id,
                "reward": reward,
                "grading_feedback": grading_result["feedback"],
                "dimension_scores": scores,
                "report_length": len(params.report),
                "num_findings": len(params.key_findings),
                "num_sources": len(params.sources_cited),
                "confidence": params.confidence,
                "query": self.config.query,
                "has_reference": self.config.has_reference
            },
            reward=reward,
            finished=True
        )

    async def _grade_report(
        self,
        report: str,
        key_findings: List[str],
        sources_cited: List[str],
        confidence: float
    ) -> Dict:
        """
        Use LLM grader with ArenaRL's 7-dimension rubric for Open-DeepResearch.

        Each dimension is evaluated separately in parallel for:
        - More focused evaluation per dimension
        - Reduced JSON parsing complexity
        - Parallel execution (7 calls via asyncio.gather)
        - No dimension anchoring bias

        Paper uses Qwen3-Max/Claude-4-Sonnet, we use gpt-5-mini for cost efficiency.

        Reference: https://arxiv.org/abs/2601.06487

        Args:
            report: Research report text
            key_findings: List of key findings
            sources_cited: List of source URLs
            confidence: Agent's confidence score

        Returns:
            Dict with keys: reward (float), feedback (str), scores (dict of 7 dimensions)
        """
        # Define all 7 dimensions with specific evaluation criteria
        dimensions = {
            "framework": "Structural completeness and logical coherence. Score 0.0-1.0.",
            "tool_usage": "Appropriateness and efficiency of tool invocations (search queries, URL fetching). Score 0.0-1.0.",
            "coverage": "Sufficiency of retrieved information - breadth of sources and aspects covered. Score 0.0-1.0.",
            "relevance": "How well the report addresses the specific research question. Score 0.0-1.0.",
            "accuracy": "Factual correctness, consistency, and plausibility of claims. Score 0.0-1.0.",
            "depth": "Analytical depth, synthesis, and reasoning coherence beyond surface-level facts. Score 0.0-1.0.",
            "clarity": "Organization, readability, and practical usability of the report. Score 0.0-1.0."
        }

        async def evaluate_dimension(dimension_name: str, criteria: str) -> tuple[str, float]:
            """Evaluate a single dimension with focused LLM call."""
            key_findings_text = "\n".join(f"- {f}" for f in key_findings)
            sources_text = "\n".join(f"- {s}" for s in sources_cited)

            prompt = f"""Evaluate this research report on ONE dimension: {dimension_name.upper()}

Research Question: {self.config.query}

Report:
{report}

Key Findings:
{key_findings_text}

Sources:
{sources_text}

Evaluation Criteria - {dimension_name}:
{criteria}

Provide:
1. A score from 0.0 to 1.0 (one decimal place)
2. Brief justification (1 sentence)

Format your response as:
Score: <0.0-1.0>
Reason: <brief justification>"""

            # Use gpt-5-mini (NO temperature parameter per migration guide)
            response = await self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            content = response.choices[0].message.content or ""

            # Parse score
            score_match = re.search(r"Score:\s*([0-9.]+)", content)
            score = float(score_match.group(1)) if score_match else 0.5

            # Clamp to valid range
            score = max(0.0, min(1.0, score))

            return dimension_name, score

        # Evaluate all 7 dimensions in parallel
        results = await asyncio.gather(
            *[evaluate_dimension(name, criteria) for name, criteria in dimensions.items()]
        )

        # Combine results
        scores = dict(results)

        # Calculate weighted reward (ArenaRL weights)
        weights = {
            "framework": 0.15,
            "tool_usage": 0.15,
            "coverage": 0.15,
            "relevance": 0.15,
            "accuracy": 0.15,
            "depth": 0.15,
            "clarity": 0.10
        }

        reward = sum(scores[dim] * weight for dim, weight in weights.items())

        return {
            "reward": max(0.0, min(1.0, reward)),
            "scores": scores,
            "feedback": f"Overall quality: {reward:.2f}/1.0"
        }


# Minimal server wrapper (8 lines)
if __name__ == "__main__":
    server = Server([OpenDeepResearch])
    server.run()
