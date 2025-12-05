"""
Pydantic models for prompt templates.

Each model corresponds to a Jinja template in the prompts/ directory,
providing type-safe validation and clear documentation of required variables.
"""
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
        min_length=2,
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
    """A single exemplary segment identified in a chapter.

    Focus: Form and function, not content themes.
    Used as output from LLM analysis to identify segments worth cataloging
    as few-shot examples.
    """
    paragraph_start: int = Field(
        ...,
        ge=0,
        description="Starting paragraph index (0-indexed)"
    )
    paragraph_end: int = Field(
        ...,
        gt=0,
        description="Ending paragraph index (exclusive, like Python slicing)"
    )
    functional_description: str = Field(
        ...,
        min_length=20,
        max_length=500,
        description="What this segment accomplishes: explains, persuades, defines, transitions, etc."
    )
    formal_description: str = Field(
        ...,
        min_length=20,
        max_length=500,
        description="How this segment is structured: syntax patterns, paragraph organization, logical flow, etc."
    )
    suggested_tags: List[str] = Field(
        ...,
        min_items=2,
        max_items=8,
        description="Descriptive tags focusing on form/function (e.g., 'clear_definition', 'parallel_structure', 'gradual_buildup')"
    )

    @field_validator('paragraph_end')
    @classmethod
    def validate_range(cls, v: int, info) -> int:
        """Ensure paragraph_end > paragraph_start."""
        if 'paragraph_start' in info.data and v <= info.data['paragraph_start']:
            raise ValueError("paragraph_end must be greater than paragraph_start")
        return v

    @field_validator('suggested_tags')
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
    """Complete analysis result: list of exemplary segments from a chapter.

    LLM should identify 10-15 segments per chapter focusing on diverse
    forms and functions. This serves as the structured output from the
    segment analysis workflow.
    """
    segments: List[ExemplarySegment] = Field(
        ...,
        min_length=10,
        max_length=15,
        description="10-15 exemplary segments demonstrating diverse forms/functions"
    )
    analysis_notes: str = Field(
        default="",
        max_length=1000,
        description="Optional: Brief notes on selection criteria or patterns observed"
    )

    @field_validator('segments')
    @classmethod
    def validate_no_overlaps(cls, v: List[ExemplarySegment]) -> List[ExemplarySegment]:
        """Check for overlapping segments (soft warning, not enforced).

        Overlaps might be intentional when demonstrating different aspects
        of the same text, so we don't fail validation, just track them.
        """
        # Sort by start for overlap detection
        sorted_segs = sorted(v, key=lambda s: s.paragraph_start)

        overlaps = []
        for i in range(len(sorted_segs) - 1):
            if sorted_segs[i].paragraph_end > sorted_segs[i+1].paragraph_start:
                overlaps.append((i, i+1))

        # Overlaps are allowed but noted
        # Future: could add logging here if needed

        return v


class ExemplarySegmentAnalysisConfig(BasePromptConfig):
    """Configuration for exemplary_segment_analysis.jinja.

    Analyzes a chapter to identify 10-15 exemplary segments worth
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
    chapter_description: str | None = Field(
        None,
        description="Optional: Brief description of chapter content/theme"
    )

    @classmethod
    def template_name(cls) -> str:
        return "exemplary_segment_analysis"
