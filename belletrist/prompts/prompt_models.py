"""
Pydantic models for prompt templates.

Each model corresponds to a Jinja template in the prompts/ directory,
providing type-safe validation and clear documentation of required variables.
"""
from typing import List, Dict, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Base Models
# =============================================================================

class BasePromptConfig(BaseModel, ABC):
    """Abstract base class for all prompt configurations."""

    @classmethod
    @abstractmethod
    def template_name(cls) -> str:
        """Return the name of the Jinja template file (without .jinja extension)."""
        pass

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Prevent accidental extra fields


# =============================================================================
# Style Retrieval: Segment Analysis Models
# =============================================================================

class ExemplarySegment(BaseModel):
    """A single exemplary passage identified in a chapter.

    Focus: Teachable craft moves demonstrating form and function.
    Used as output from LLM analysis to identify passages worth cataloging
    as few-shot examples.
    """
    text: str = Field(
        ...,
        min_length=50,
        description="The full extracted passage demonstrating the craft move"
    )
    craft_move: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Short, specific name for what this passage demonstrates (e.g., 'concessive_pivot', 'rhythmic_clincher')"
    )
    teaching_note: str = Field(
        ...,
        min_length=50,
        max_length=800,
        description="2-4 sentences explaining what makes this passage exemplary. Write as if briefing another writing teacher."
    )
    tags: List[str] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="2-5 lowercase tags for retrieval: 1-2 canonical tags (Tier 1) + 0-3 author-specific tags (Tier 2)"
    )

    @field_validator('craft_move')
    @classmethod
    def validate_craft_move(cls, v: str) -> str:
        """Ensure craft_move uses underscores, is lowercase."""
        cleaned = v.lower().strip().replace(' ', '_').replace('-', '_')
        if not cleaned:
            raise ValueError("craft_move cannot be empty")
        return cleaned

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are lowercase with underscores, normalized."""
        cleaned = []
        for tag in v:
            # Convert to lowercase, replace spaces/hyphens with underscores
            cleaned_tag = tag.lower().strip().replace(' ', '_').replace('-', '_')
            if cleaned_tag:
                cleaned.append(cleaned_tag)

        if len(cleaned) < 2:
            raise ValueError("At least 2 valid tags required after normalization")

        return cleaned


class ExemplarySegmentAnalysis(BaseModel):
    """Complete analysis result: list of exemplary passages from a chapter.

    LLM identifies passages demonstrating teachable craft moves.
    This serves as the structured output from the segment analysis workflow.
    """
    passages: List[ExemplarySegment] = Field(
        ...,
        min_length=5,
        max_length=20,
        description="List of exemplary passages demonstrating diverse craft moves"
    )
    overall_observations: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional: Patterns noticed across selections that characterize the author's style broadly"
    )


class ExemplarySegmentAnalysisConfig(BasePromptConfig):
    """Configuration for segment_style_analysis.jinja.

    Analyzes a chapter to identify exemplary segments worth
    cataloging as few-shot examples. Focus is on form/function, not
    content themes.
    """

    chapter_text: str = Field(
        ...,
        min_length=1000,
        description="Full chapter text to analyze"
    )
    file_name: str = Field(
        ...,
        min_length=1,
        description="Source file name for context"
    )
    num_segments: int = Field(
        default=12,
        ge=5,
        le=20,
        description="Number of exemplary segments to identify (5-20)"
    )
    chapter_description: str | None = Field(
        None,
        description="Optional: Brief description of chapter content/theme"
    )
    existing_tier2_tags: List[str] = Field(
        default_factory=list,
        description="Author-specific tags already in catalog (Tier 2 only; canonical tags excluded)"
    )
    canonical_tags_formatted: List[dict] = Field(
        default_factory=list,
        description="Formatted canonical tags for template injection (auto-populated from canonical_tags.py)"
    )

    @classmethod
    def template_name(cls) -> str:
        return "segment_style_analysis"


# =============================================================================
# Style Evaluation Models
# =============================================================================

class StyleFlatteningConfig(BasePromptConfig):
    """Configuration for style_flattening.jinja - moderate content extraction.

    Extracts semantic and argumentative content from text while removing
    all stylistic elements, creating a style-neutral summary suitable for
    reconstruction experiments. Produces prose-form output at ~70-90% of
    original length.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to extract content from"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_flattening"


class StyleNeutralizationConfig(BasePromptConfig):
    """Configuration for style_neutralization.jinja - neutral journalistic rewrite.

    Rewrites text in bland, straightforward journalistic prose while preserving
    the complete rhetorical and argumentative structure. Maintains concessions,
    qualifications, logical connectors, and emphasis patterns, but removes
    distinctive stylistic choices.

    Different from StyleFlatteningConfig (which extracts/summarizes).
    This produces a full-length neutral rewrite (~80-100% of original length)
    in plain, functional prose suitable for later stylistic reconstruction.

    Think: AP style or plain academic prose - clear and functional but
    stylistically unmarked.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to rewrite in neutral style"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_neutralization"


class StyleReconstructionGenericConfig(BasePromptConfig):
    """Configuration for style_reconstruction_generic.jinja - baseline reconstruction.

    Generic baseline: expands content with standard "write clearly" instructions.
    """

    content_summary: str = Field(
        ...,
        min_length=1,
        description="Style-flattened content summary to expand"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_reconstruction_generic"


class StyleReconstructionFewShotConfig(BasePromptConfig):
    """Configuration for style_reconstruction_fewshot.jinja - few-shot learning.

    Provides example texts to guide style reconstruction through implicit learning.
    """

    content_summary: str = Field(
        ...,
        min_length=1,
        description="Style-flattened content summary to expand"
    )
    few_shot_examples: list[str] = Field(
        ...,
        min_length=1,
        description="2-3 example texts demonstrating the target style"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_reconstruction_fewshot"


class StyleReconstructionAuthorConfig(BasePromptConfig):
    """Configuration for style_reconstruction_author.jinja - author name prompting.

    Leverages model's implicit knowledge of author style through name prompting.
    """

    content_summary: str = Field(
        ...,
        min_length=1,
        description="Style-flattened content summary to expand"
    )
    author_name: str = Field(
        ...,
        min_length=1,
        description="Name of author whose style to emulate"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_reconstruction_author"


class StyleJudgeComparativeConfig(BasePromptConfig):
    """Configuration for style_judge_comparative.jinja - blind comparative ranking.

    Provides the original text and 4 anonymously labeled reconstructions (A, B, C, D)
    for comparative blind evaluation. The judge ranks all 4 from 1-4 based on
    stylistic similarity to the original.
    """

    original_text: str = Field(
        ...,
        min_length=1,
        description="The original gold standard text"
    )
    reconstruction_text_a: str = Field(
        ...,
        min_length=1,
        description="Reconstruction labeled as Text A (anonymous)"
    )
    reconstruction_text_b: str = Field(
        ...,
        min_length=1,
        description="Reconstruction labeled as Text B (anonymous)"
    )
    reconstruction_text_c: str = Field(
        ...,
        min_length=1,
        description="Reconstruction labeled as Text C (anonymous)"
    )
    reconstruction_text_d: str = Field(
        ...,
        min_length=1,
        description="Reconstruction labeled as Text D (anonymous)"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_judge_comparative"


# =============================================================================
# Holistic Strategic Retrieval Models
# =============================================================================

class StrategicRetrievalPlan(BaseModel):
    """Strategic plan for holistic style rewriting.

    Identifies rhetorical situation and selects craft palette for entire text
    (not per-paragraph). Planner analyzes what the text needs overall and
    selects 6-12 tags covering functional categories: openers, builders,
    closers, texture/stance.

    This enables lean rewriting closer to vanilla few-shot (which performs best)
    while maintaining strategic example selection.
    """
    structural_diagnosis: str = Field(
        ...,
        min_length=50,
        max_length=300,
        description="1-2 sentences analyzing what the text needs rhetorically (e.g., 'Claim-based argument needing grounding and concession')"
    )
    selected_tags: List[str] = Field(
        ...,
        min_length=6,
        max_length=12,
        description="6-12 tags covering craft palette: openers (1-2), builders (3-4), closers (1-2), texture/stance (2-3)"
    )
    target_example_count: int = Field(
        default=10,
        ge=8,
        le=10,
        description="Number of examples to retrieve (8-10)"
    )

    @field_validator('selected_tags')
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are lowercase with underscores."""
        return [tag.lower().strip().replace(' ', '_').replace('-', '_') for tag in v]


class StrategicRetrievalPlannerConfig(BasePromptConfig):
    """Configuration for strategic_retrieval_planner.jinja.

    Analyzes entire flattened text holistically to identify rhetorical situation
    and select strategic craft palette (not per-paragraph guidance).

    Planner asks: "What does this text need rhetorically?" and selects 6-12 tags
    covering functional categories. This enables retrieving 8-10 strategic examples
    (vs 30+ per-paragraph examples) for lean rewriting.
    """
    flattened_text: str = Field(
        ...,
        min_length=100,
        description="Complete style-flattened text to analyze holistically"
    )
    tier2_tags: List[str] = Field(
        default_factory=list,
        description="Author-specific tags from catalog (Tier 2 only)"
    )
    canonical_tags_formatted: List[dict] = Field(
        default_factory=list,
        description="Formatted canonical tags (auto-populated from canonical_tags.py)"
    )
    creative_latitude: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate",
        description="How much creative freedom in style enhancement"
    )

    @classmethod
    def template_name(cls) -> str:
        return "strategic_retrieval_planner"


class HolisticStyledRewriteConfig(BasePromptConfig):
    """Configuration for holistic_styled_rewrite.jinja.

    Lean rewriter template closer to vanilla few-shot structure (which ranks best).
    Shows flattened text, structural diagnosis, and 8-10 strategic examples.
    No paragraph-level scaffolding or forced structure.

    This tests the hypothesis that paragraph-level scaffolding (not teaching notes)
    causes poor performance in agent_fewshot (ranks 3.10 vs fewshot's 1.30).
    """
    flattened_text: str = Field(
        ...,
        min_length=100,
        description="Style-flattened content to expand"
    )
    plan: StrategicRetrievalPlan = Field(
        ...,
        description="Strategic plan with diagnosis and selected tags"
    )
    retrieved_examples: List[dict] = Field(
        ...,
        min_length=8,
        max_length=10,
        description="8-10 strategic examples with craft_move, teaching_note, text"
    )
    include_teaching_notes: bool = Field(
        default=True,
        description="Whether to show teaching notes (allows A/B testing)"
    )

    @classmethod
    def template_name(cls) -> str:
        return "holistic_styled_rewrite"


class StatisticalFewShotRewriteConfig(BasePromptConfig):
    """Configuration for statistical_fewshot_rewrite.jinja.

    Pure few-shot template with statistically selected examples.
    No planning, no diagnosis, no craft_move labels.
    Mimics vanilla few-shot structure but uses catalog examples.

    This variant removes the planning phase entirely and uses purely
    statistical selection to choose examples representing common tags
    with diversity. Aims to match or exceed vanilla few-shot performance
    (rank 1.30) by following its simple structure.
    """
    content_summary: str = Field(
        ...,
        min_length=100,
        description="Neutral text to enhance"
    )
    few_shot_examples: List[dict] = Field(
        ...,
        min_length=1,
        description="List of examples, each with 'teaching_note' and 'text' keys"
    )

    @classmethod
    def template_name(cls) -> str:
        return "statistical_fewshot_rewrite"
