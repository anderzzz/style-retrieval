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


class SpecialistAnalystConfig(BasePromptConfig, ABC):
    """
    Base class for specialist analyst templates.

    All specialist analysts have 4 optional boolean sections that can be
    enabled/disabled. By default, all sections are enabled.
    """

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        """Return a brief description of this analyst's focus area."""
        pass

    @classmethod
    @abstractmethod
    def display_name(cls) -> str:
        """Return the formatted display name for this analyst."""
        pass


# =============================================================================
# Foundational Template Models
# =============================================================================

class BasicPromptConfig(BasePromptConfig):
    """Configuration for basic_prompt.jinja - generic role-based prompts."""

    role: str = Field(
        ...,
        min_length=1,
        description="The role or persona for the AI to adopt"
    )
    context: str | None = Field(
        None,
        description="Optional context information to provide background"
    )
    question: str = Field(
        ...,
        min_length=1,
        description="The main question or task to address"
    )
    instructions: list[str] = Field(
        default_factory=list,
        description="Optional list of specific instructions to follow"
    )

    @classmethod
    def template_name(cls) -> str:
        return "basic_prompt"


class PreambleTextConfig(BasePromptConfig):
    """Configuration for preamble_text.jinja - text passage container."""

    text_to_analyze: str = Field(
        ...,
        min_length=1,
        description="The prose text to be analyzed"
    )

    @classmethod
    def template_name(cls) -> str:
        return "preamble_text"


class PreambleInstructionConfig(BasePromptConfig):
    """
    Configuration for preamble_instruction.jinja - static preamble.

    This template has no variables; it's a static instructional preamble
    explaining the analytical task.
    """

    @classmethod
    def template_name(cls) -> str:
        return "preamble_instruction"


# =============================================================================
# Specialist Analyst Models
# =============================================================================

class SyntacticianConfig(SpecialistAnalystConfig):
    """Configuration for syntactician.jinja - syntax and sentence structure analysis."""

    include_sentence_structures: bool = Field(
        True,
        description="Analyze sentence length, types, and variety"
    )
    include_clause_architecture: bool = Field(
        True,
        description="Analyze clause relationships and dependencies"
    )
    include_grammatical_features: bool = Field(
        True,
        description="Analyze voice, mood, tense, and aspect"
    )
    include_functional_observations: bool = Field(
        True,
        description="Analyze how syntax serves meaning and effect"
    )

    @classmethod
    def template_name(cls) -> str:
        return "syntactician"

    @classmethod
    def description(cls) -> str:
        return "Sentence structure, clause architecture, grammatical patterns"

    @classmethod
    def display_name(cls) -> str:
        return "Syntactician"


class LexicologistConfig(SpecialistAnalystConfig):
    """Configuration for lexicologist.jinja - vocabulary and diction analysis."""

    include_lexical_register: bool = Field(
        True,
        description="Analyze formality and word choice patterns"
    )
    include_semantic_fields: bool = Field(
        True,
        description="Analyze word meanings and conceptual groupings"
    )
    include_precision_analysis: bool = Field(
        True,
        description="Analyze specificity and exactness of word choice"
    )
    include_clarity_mechanisms: bool = Field(
        True,
        description="Analyze how vocabulary achieves clarity"
    )

    @classmethod
    def template_name(cls) -> str:
        return "lexicologist"

    @classmethod
    def description(cls) -> str:
        return "Word choice, register, etymology, semantic fields"

    @classmethod
    def display_name(cls) -> str:
        return "Lexicologist"


class InformationArchitectConfig(SpecialistAnalystConfig):
    """Configuration for information_architect.jinja - information structure analysis."""

    include_paragraph_architecture: bool = Field(
        True,
        description="Analyze paragraph structure and organization"
    )
    include_coherence_mechanisms: bool = Field(
        True,
        description="Analyze how ideas connect and relate"
    )
    include_logical_progression: bool = Field(
        True,
        description="Analyze the sequence and development of ideas"
    )
    include_transitions: bool = Field(
        True,
        description="Analyze transition techniques and flow"
    )

    @classmethod
    def template_name(cls) -> str:
        return "information_architect"

    @classmethod
    def description(cls) -> str:
        return "Logical flow, coherence, information structure"

    @classmethod
    def display_name(cls) -> str:
        return "Information Architect"


class RhetoricianConfig(SpecialistAnalystConfig):
    """Configuration for rhetorician.jinja - rhetorical strategy analysis."""

    include_writer_position: bool = Field(
        True,
        description="Analyze the writer's stance and voice"
    )
    include_reader_positioning: bool = Field(
        True,
        description="Analyze how the text positions the reader"
    )
    include_persuasive_techniques: bool = Field(
        True,
        description="Analyze persuasive devices and appeals"
    )
    include_argumentative_moves: bool = Field(
        True,
        description="Analyze argumentative structure and logic"
    )

    @classmethod
    def template_name(cls) -> str:
        return "rhetorician"

    @classmethod
    def description(cls) -> str:
        return "Persuasive strategy, tone, reader positioning"

    @classmethod
    def display_name(cls) -> str:
        return "Rhetorician"


class EfficiencyAuditorConfig(SpecialistAnalystConfig):
    """Configuration for efficiency_auditor.jinja - economy and compression analysis."""

    include_word_economy: bool = Field(
        True,
        description="Analyze conciseness and word efficiency"
    )
    include_structural_efficiency: bool = Field(
        True,
        description="Analyze sentence and paragraph efficiency"
    )
    include_density_analysis: bool = Field(
        True,
        description="Analyze information density and payload"
    )
    include_subtraction_test: bool = Field(
        True,
        description="Analyze necessity of each element"
    )

    @classmethod
    def template_name(cls) -> str:
        return "efficiency_auditor"

    @classmethod
    def description(cls) -> str:
        return "Economy, necessity, compression"

    @classmethod
    def display_name(cls) -> str:
        return "Efficiency Auditor"


# =============================================================================
# Integration & Synthesis Models
# =============================================================================

class CrossPerspectiveIntegratorConfig(BasePromptConfig):
    """
    Configuration for cross_perspective_integrator.jinja - cross-perspective integration.

    This template integrates multiple specialist analyses of a single text
    to identify unified patterns. Requires at least 2 analysts.

    The analysts dict should have structure:
    {
        'analyst_key': {
            'analysis': 'The full analysis text from this analyst',
            'analyst_descr_short': 'Brief description of analyst focus'
        },
        ...
    }
    """

    original_text: str = Field(
        ...,
        min_length=1,
        description="The original text being analyzed"
    )
    analysts: dict[str, dict[str, str]] = Field(
        ...,
        description="Dictionary mapping analyst keys to their analysis and description"
    )

    @field_validator('analysts')
    @classmethod
    def validate_analysts(cls, v: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        """Validate that at least 2 analysts are provided with required keys."""
        if len(v) < 2:
            raise ValueError("At least 2 analysts are required for cross-perspective integration")

        required_keys = {'analysis', 'analyst_descr_short'}
        for analyst_key, analyst_data in v.items():
            missing_keys = required_keys - set(analyst_data.keys())
            if missing_keys:
                raise ValueError(
                    f"Analyst '{analyst_key}' missing required keys: {missing_keys}. "
                    f"Each analyst must have 'analysis' and 'analyst_descr_short'."
                )
            if not analyst_data['analysis'].strip():
                raise ValueError(f"Analyst '{analyst_key}' has empty analysis")
            if not analyst_data['analyst_descr_short'].strip():
                raise ValueError(f"Analyst '{analyst_key}' has empty description")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "cross_perspective_integrator"

    @classmethod
    def analyst_name(cls) -> str:
        """Return the analyst identifier for storing results in ResultStore."""
        return "cross_perspective_integrator"


class CrossTextSynthesizerConfig(BasePromptConfig):
    """
    Configuration for cross_text_synthesizer.jinja - cross-text synthesis.

    This template synthesizes patterns across multiple text analyses
    to extract generalizable principles. Requires at least 2 integrated analyses.

    The integrated_analyses dict should map sample IDs to their cross-perspective
    integration outputs:
    {
        'sample_001': 'The integrated analysis for sample 001...',
        'sample_002': 'The integrated analysis for sample 002...',
        ...
    }
    """

    integrated_analyses: dict[str, str] = Field(
        ...,
        description="Mapping of sample IDs to their cross-perspective integration analyses"
    )

    @field_validator('integrated_analyses')
    @classmethod
    def validate_integrated_analyses(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate that at least 2 integrated analyses are provided."""
        if len(v) < 2:
            raise ValueError(
                "At least 2 integrated analyses are required for cross-text synthesis. "
                f"Got {len(v)}."
            )

        for sample_id, analysis in v.items():
            if not analysis.strip():
                raise ValueError(f"Integrated analysis for '{sample_id}' is empty")

        return v

    @classmethod
    def template_name(cls) -> str:
        return "cross_text_synthesizer"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return the synthesis type identifier for storing in ResultStore."""
        return "cross_text_synthesis"


class SynthesizerOfPrinciplesConfig(BasePromptConfig):
    """
    Configuration for synthesizer_of_principles.jinja - prescriptive guide generation.

    This template converts descriptive pattern analyses into actionable
    writing principles and style guidelines.
    """

    synthesis_document: str = Field(
        ...,
        min_length=1,
        description="The complete Stage 2 cross-text synthesis document"
    )

    @classmethod
    def template_name(cls) -> str:
        return "synthesizer_of_principles"

    @classmethod
    def synthesis_type(cls) -> str:
        """Return the synthesis type identifier for storing in ResultStore."""
        return "principles_guide"


# =============================================================================
# Style Evaluation Models
# =============================================================================

class StyleFlatteningConfig(BasePromptConfig):
    """
    Configuration for style_flattening.jinja - moderate content extraction.

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


class StyleFlatteningAggressiveConfig(BasePromptConfig):
    """
    Configuration for style_flattening_aggressive.jinja - aggressive compression.

    Extracts propositional content as a bare logical skeleton in outline format.
    Uses telegraphic language with no transitions or rhetoric. Aims for 30-50%
    of original length.

    More aggressive than StyleFlatteningConfig - better for testing whether
    reconstruction methods can truly rebuild style from minimal content.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The text to compress into outline format"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_flattening_aggressive"


class StyleNeutralizationConfig(BasePromptConfig):
    """
    Configuration for style_neutralization.jinja - neutral journalistic rewrite.

    Rewrites text in bland, straightforward journalistic prose while preserving
    the complete rhetorical and argumentative structure. Maintains concessions,
    qualifications, logical connectors, and emphasis patterns, but removes
    distinctive stylistic choices.

    Different from StyleFlatteningConfig (which extracts/summarizes) and
    StyleFlatteningAggressiveConfig (which compresses). This produces a
    full-length neutral rewrite (~80-100% of original length) in plain,
    functional prose suitable for later stylistic reconstruction.

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
    """
    Configuration for style_reconstruction_generic.jinja - baseline reconstruction.

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
    """
    Configuration for style_reconstruction_fewshot.jinja - few-shot learning.

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
    """
    Configuration for style_reconstruction_author.jinja - author name prompting.

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


class StyleReconstructionInstructionsConfig(BasePromptConfig):
    """
    Configuration for style_reconstruction_instructions.jinja - derived instructions.

    Applies explicitly derived style instructions from the synthesis pipeline.

    Note: If style_instructions contains YAML front-matter (delimited by ---),
    it will be automatically stripped, leaving only the instructions content.
    """

    content_summary: str = Field(
        ...,
        min_length=1,
        description="Style-flattened content summary to expand"
    )
    style_instructions: str = Field(
        ...,
        min_length=1,
        description="Derived style principles from SynthesizerOfPrinciplesConfig"
    )

    @field_validator('style_instructions')
    @classmethod
    def strip_yaml_frontmatter(cls, v: str) -> str:
        """
        Strip YAML front-matter if present.

        Front-matter is identified by starting with '---' and ending with '---',
        as exported by ResultStore.export_synthesis().
        """
        v = v.strip()

        # Check if starts with YAML front-matter delimiter
        if v.startswith('---\n') or v.startswith('---\r\n'):
            # Find the closing delimiter
            lines = v.split('\n')
            end_index = None

            for i, line in enumerate(lines[1:], start=1):  # Skip first ---
                if line.strip() == '---':
                    end_index = i
                    break

            if end_index is not None:
                # Return everything after the closing ---
                return '\n'.join(lines[end_index + 1:]).strip()

        # No front-matter found, return as-is
        return v

    @classmethod
    def template_name(cls) -> str:
        return "style_reconstruction_instructions"


class StyleJudgeComparativeConfig(BasePromptConfig):
    """
    Configuration for style_judge_comparative.jinja - blind comparative ranking.

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


class StyleJudgeComparative5WayConfig(BasePromptConfig):
    """
    Configuration for style_judge_comparative_5way.jinja - blind comparative ranking with 5 texts.

    Provides the original text and 5 anonymously labeled texts (A, B, C, D, E)
    for comparative blind evaluation. The judge ranks all 5 from 1-5 based on
    stylistic similarity to the original.

    Use case: Include the original text itself as one of the 5 options to test
    whether the judge can reliably identify it (adversarial validation).
    """

    original_text: str = Field(
        ...,
        min_length=1,
        description="The original gold standard text"
    )
    reconstruction_text_a: str = Field(
        ...,
        min_length=1,
        description="Text labeled as A (anonymous)"
    )
    reconstruction_text_b: str = Field(
        ...,
        min_length=1,
        description="Text labeled as B (anonymous)"
    )
    reconstruction_text_c: str = Field(
        ...,
        min_length=1,
        description="Text labeled as C (anonymous)"
    )
    reconstruction_text_d: str = Field(
        ...,
        min_length=1,
        description="Text labeled as D (anonymous)"
    )
    reconstruction_text_e: str = Field(
        ...,
        min_length=1,
        description="Text labeled as E (anonymous)"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_judge_comparative_5way"


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
# Style Rewriting: Planning and Execution Models
# =============================================================================

class ParagraphPlan(BaseModel):
    """Plan for rewriting a single paragraph.

    Contains the original text, its rhetorical function, and craft move guidance
    for stylistic rewriting.

    Note: Field limits are forgiving guardrails, not strict targets.
    Prompts should guide LLMs toward concise output, but modest overruns are acceptable.
    """
    paragraph_id: int = Field(..., ge=0, description="Paragraph index (0-based)")
    original_text: str = Field(..., min_length=10, description="Original paragraph text")
    function: str = Field(
        ..., min_length=10, max_length=300,
        description="Rhetorical function (e.g., 'introduces_concept', 'counterargument', 'conclusion')"
    )
    craft_move: str = Field(
        ..., min_length=3, max_length=100,
        description="Primary craft move to apply (e.g., 'concessive_opening', 'parallel_structure')"
    )
    craft_tags: List[str] = Field(
        ..., min_length=1, max_length=5,
        description="Tags for retrieving relevant examples from catalog (1-5 tags)"
    )
    guidance: str = Field(
        ..., min_length=20, max_length=500,
        description="Brief instruction on how to apply the craft move (aim for 2-3 sentences, ~300 chars)"
    )

    @field_validator('craft_move')
    @classmethod
    def normalize_craft_move(cls, v: str) -> str:
        """Ensure craft_move uses underscores, is lowercase."""
        return v.lower().strip().replace(' ', '_').replace('-', '_')

    @field_validator('craft_tags')
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Ensure tags are lowercase with underscores, normalized."""
        return [tag.lower().strip().replace(' ', '_').replace('-', '_') for tag in v]


class StyleRewritePlan(BaseModel):
    """Complete rewriting plan with paragraph-level guidance.

    Output from planning agent that analyzes flattened text and prescribes
    craft moves for each paragraph.

    Note: Field limits are forgiving guardrails, not strict targets.
    Prompts should guide toward concise output, but modest overruns are acceptable.
    """
    paragraphs: List[ParagraphPlan] = Field(
        ..., min_length=1, max_length=50,
        description="Paragraph-level rewriting plans (1-50 paragraphs)"
    )
    overall_strategy: str = Field(
        ..., min_length=30, max_length=800,
        description="High-level description of rewrite approach (aim for ~300-500 chars)"
    )


class StyleRewritePlannerConfig(BasePromptConfig):
    """Configuration for style_rewrite_planner.jinja.

    Planning agent analyzes flattened (style-sparse) text and creates a
    paragraph-by-paragraph rewrite plan with craft move assignments.
    """

    flattened_text: str = Field(
        ..., min_length=100,
        description="Style-flattened input text to analyze"
    )
    tier2_tags: List[str] = Field(
        default_factory=list,
        description="Author-specific tags available in catalog (Tier 2 only; canonical tags excluded)"
    )
    canonical_tags_formatted: List[dict] = Field(
        default_factory=list,
        description="Formatted canonical tags for template injection (auto-populated from canonical_tags.py)"
    )
    target_style_description: str = Field(
        default="balanced, lucid, rhythmically varied",
        description="Brief description of target style"
    )
    creative_latitude: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate",
        description="How much creative freedom in suggesting craft moves"
    )

    @classmethod
    def template_name(cls) -> str:
        return "style_rewrite_planner"


class StyledRewriteConfig(BasePromptConfig):
    """Configuration for styled_rewrite.jinja.

    Rewriting agent receives plan with retrieved examples and generates
    styled output paragraph-by-paragraph.
    """

    plan: StyleRewritePlan = Field(
        ...,
        description="Complete rewriting plan from planning agent"
    )
    retrieved_examples: Dict[int, List[dict]] = Field(
        ...,
        description="Map of paragraph_id → list of retrieved example dicts"
    )

    @classmethod
    def template_name(cls) -> str:
        return "styled_rewrite"


class StyledRewriteNoCraftNotesConfig(BasePromptConfig):
    """Configuration for styled_rewrite_no_annotations.jinja.

    Like StyledRewriteConfig, but omits craft metadata (function, craft_move,
    guidance, teaching_note, tags) to test if simpler prompts perform better.

    Shows only: original paragraph text → example passages (raw text only).
    """

    plan: StyleRewritePlan = Field(
        ...,
        description="Complete rewriting plan from planning agent"
    )
    retrieved_examples: Dict[int, List[dict]] = Field(
        ...,
        description="Map of paragraph_id → list of retrieved example dicts"
    )

    @classmethod
    def template_name(cls) -> str:
        return "styled_rewrite_no_annotations"


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
