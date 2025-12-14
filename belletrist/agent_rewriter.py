"""
Clean wrapper for agent-based style rewriting.

Extracts the 3-phase workflow from runs/style_rewrite.py with ALL debugging output removed.
This provides a clean interface suitable for evaluation experiments.

Workflow:
1. Planning Agent: Analyzes flattened text, creates paragraph-level plans
2. Retrieval: Searches catalog for craft-tagged examples using heuristics
3. Rewriting Agent: Generates styled output using retrieved examples

Returns ONLY the styled text (no debugging, no progress messages, no prompt dumps).
"""
from typing import List, Dict
from pathlib import Path

from belletrist import LLM, PromptMaker, SegmentStore
from belletrist.prompts import (
    StyleRewritePlannerConfig,
    StyledRewriteConfig,
    StyledRewriteNoCraftNotesConfig,
    StyleRewritePlan
)
from belletrist.segment_store import SegmentRecord


def select_examples_with_heuristics(
    candidates: List[SegmentRecord],
    max_examples: int = 3
) -> List[SegmentRecord]:
    """Select best examples using deterministic heuristics.

    Ranking criteria:
    1. Length (prefer shorter)
    2. Tag specificity (prefer fewer tags)
    3. Source diversity (prefer different files)

    Args:
        candidates: List of candidate SegmentRecords
        max_examples: Maximum examples to return

    Returns:
        Ranked list of top examples
    """
    if not candidates:
        return []

    scored = []
    for segment in candidates:
        score = 0

        # 1. Prefer shorter examples (easier to learn from)
        length = len(segment.text)
        if length < 500:
            score += 3
        elif length < 1000:
            score += 2
        else:
            score += 1

        # 2. Prefer focused examples (fewer tags = more specific)
        num_tags = len(segment.tags)
        if num_tags <= 3:
            score += 3
        elif num_tags <= 5:
            score += 2
        else:
            score += 1

        scored.append((score, segment))

    # Sort by score descending
    scored.sort(reverse=True, key=lambda x: x[0])

    # Take top N, ensuring source diversity
    selected = []
    seen_files = set()

    for score, segment in scored:
        # Prefer diversity (different source files)
        if len(selected) < max_examples:
            selected.append(segment)
            seen_files.add(segment.file_name)
        elif segment.file_name not in seen_files and len(selected) < max_examples * 2:
            # Allow duplicates from same file only if we don't have enough
            selected.append(segment)

    return selected[:max_examples]


def _plan_rewrite(
    flattened_text: str,
    available_tags: List[str],
    llm: LLM,
    prompt_maker: PromptMaker,
    creative_latitude: str = "moderate"
) -> StyleRewritePlan:
    """Phase 1: Planning agent analyzes text and creates rewrite plan.

    Args:
        flattened_text: Style-sparse input text
        available_tags: All tags from segment catalog (will be filtered internally)
        llm: LLM instance (should use planning temperature ~0.5)
        prompt_maker: PromptMaker instance
        creative_latitude: "conservative", "moderate", or "aggressive"

    Returns:
        StyleRewritePlan with paragraph-level guidance

    Raises:
        ValueError: If schema validation fails
    """
    from belletrist.prompts.canonical_tags import get_all_canonical_tags, format_for_jinja

    # Filter out canonical tags to get only Tier 2
    canonical_tag_set = get_all_canonical_tags()
    tier2_tags = [tag for tag in available_tags if tag not in canonical_tag_set]

    # Get formatted canonical tags for template injection
    canonical_tags_formatted = format_for_jinja()

    # Create config
    config = StyleRewritePlannerConfig(
        flattened_text=flattened_text,
        tier2_tags=tier2_tags,
        canonical_tags_formatted=canonical_tags_formatted,
        creative_latitude=creative_latitude
    )

    # Render prompt
    prompt = prompt_maker.render(config)

    # Call LLM with schema (no debugging output)
    response = llm.complete_with_schema(
        prompt=prompt,
        schema_model=StyleRewritePlan,
        system="You are a JSON API that returns structured data. Always respond with valid JSON matching the requested schema. Never include explanatory text or wrapper objects."
        # Uses strict=True by default; falls back to json_object mode if provider doesn't support it
    )

    plan: StyleRewritePlan = response.content
    return plan


def _retrieve_examples(
    plan: StyleRewritePlan,
    store: SegmentStore,
    num_examples: int = 3
) -> Dict[int, List[dict]]:
    """Phase 2: Retrieve relevant examples for each paragraph using heuristics.

    Args:
        plan: StyleRewritePlan from planning agent
        store: SegmentStore instance
        num_examples: Number of examples to retrieve per paragraph

    Returns:
        Dict mapping paragraph_id -> list of example dicts
    """
    examples_by_paragraph = {}

    for para_plan in plan.paragraphs:
        para_id = para_plan.paragraph_id

        # Collect candidates from all tags
        candidates = []
        for tag in para_plan.craft_tags:
            tag_results = store.search_by_tag(tag, exact_match=True)
            candidates.extend(tag_results)

        # Deduplicate by segment_id
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.segment_id not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate.segment_id)

        # Apply selection heuristics
        selected = select_examples_with_heuristics(
            unique_candidates,
            max_examples=num_examples
        )

        # Convert to dicts for template
        examples_by_paragraph[para_id] = [
            {
                'segment_id': seg.segment_id,
                'craft_move': seg.craft_move,
                'teaching_note': seg.teaching_note,
                'tags': seg.tags,
                'text': seg.text
            }
            for seg in selected
        ]

    return examples_by_paragraph


def _rewrite_with_style(
    plan: StyleRewritePlan,
    examples: Dict[int, List[dict]],
    llm: LLM,
    prompt_maker: PromptMaker
) -> str:
    """Phase 3: Rewriting agent generates styled output.

    Args:
        plan: StyleRewritePlan with paragraph guidance
        examples: Retrieved examples by paragraph_id
        llm: LLM instance (should use rewriting temperature ~0.7)
        prompt_maker: PromptMaker instance

    Returns:
        Styled rewritten text (ONLY the text, no metadata)
    """
    # Create config
    config = StyledRewriteConfig(
        plan=plan,
        retrieved_examples=examples
    )

    # Render prompt
    prompt = prompt_maker.render(config)

    # Call LLM (text output, not schema)
    # NO DEBUGGING OUTPUT HERE (removed print(f"AAA\n{prompt}") that was on line 307)
    response = llm.complete(prompt)

    styled_text = response.content
    return styled_text


def agent_rewrite(
    flattened_content: str,
    segment_store: SegmentStore,
    planning_llm: LLM,
    rewriting_llm: LLM,
    prompt_maker: PromptMaker,
    creative_latitude: str = "moderate",
    num_examples_per_paragraph: int = 3
) -> str:
    """
    Clean wrapper for agent-based style rewriting.

    Orchestrates 3-phase workflow:
    1. Planning: Analyze flattened text, create paragraph-level plans
    2. Retrieval: Search catalog for craft-tagged examples
    3. Rewriting: Generate styled output using retrieved examples

    Args:
        flattened_content: Style-flattened input text to rewrite
        segment_store: SegmentStore instance with catalog of examples
        planning_llm: LLM instance for planning (recommended temp=0.5)
        rewriting_llm: LLM instance for rewriting (recommended temp=0.7)
        prompt_maker: PromptMaker instance for template rendering
        creative_latitude: "conservative", "moderate", or "aggressive"
        num_examples_per_paragraph: Number of examples to retrieve per paragraph

    Returns:
        str: Styled rewritten text (ONLY the text, no debugging output)

    Raises:
        ValueError: If planning fails schema validation or catalog search fails

    Example:
        >>> from belletrist import LLM, LLMConfig, PromptMaker, SegmentStore
        >>> from belletrist.agent_rewriter import agent_rewrite
        >>>
        >>> planning_llm = LLM(LLMConfig(model="gpt-4", api_key=key, temperature=0.5))
        >>> rewriting_llm = LLM(LLMConfig(model="gpt-4", api_key=key, temperature=0.7))
        >>>
        >>> with SegmentStore("segments.db") as store:
        >>>     styled_text = agent_rewrite(
        >>>         flattened_content="Democracy has flaws. But it remains best.",
        >>>         segment_store=store,
        >>>         planning_llm=planning_llm,
        >>>         rewriting_llm=rewriting_llm,
        >>>         prompt_maker=PromptMaker()
        >>>     )
        >>> print(styled_text)  # Clean output, no debugging
    """
    # Phase 1: Planning (silently)
    available_tags = list(segment_store.list_all_tags().keys())
    plan = _plan_rewrite(
        flattened_text=flattened_content,
        available_tags=available_tags,
        llm=planning_llm,
        prompt_maker=prompt_maker,
        creative_latitude=creative_latitude
    )

    # Phase 2: Retrieval (silently)
    examples = _retrieve_examples(
        plan=plan,
        store=segment_store,
        num_examples=num_examples_per_paragraph
    )

    # Phase 3: Rewriting (silently)
    styled_text = _rewrite_with_style(
        plan=plan,
        examples=examples,
        llm=rewriting_llm,
        prompt_maker=prompt_maker
    )

    # Return ONLY the styled text, no debugging artifacts
    return styled_text


def _rewrite_with_style_no_annotations(
    plan: StyleRewritePlan,
    examples: Dict[int, List[dict]],
    llm: LLM,
    prompt_maker: PromptMaker
) -> str:
    """Phase 3: Rewriting agent generates styled output WITHOUT craft annotations.

    Like _rewrite_with_style, but uses stripped-down template showing only
    original paragraph text and raw example passages (no craft metadata).

    Args:
        plan: StyleRewritePlan with paragraph guidance
        examples: Retrieved examples by paragraph_id
        llm: LLM instance (should use rewriting temperature ~0.7)
        prompt_maker: PromptMaker instance

    Returns:
        Styled rewritten text (ONLY the text, no metadata)
    """
    # Create config (uses no-annotations template)
    config = StyledRewriteNoCraftNotesConfig(
        plan=plan,
        retrieved_examples=examples
    )

    # Render prompt
    prompt = prompt_maker.render(config)

    # Call LLM (text output, not schema)
    response = llm.complete(prompt)

    styled_text = response.content
    return styled_text


def agent_rewrite_no_annotations(
    flattened_content: str,
    segment_store: SegmentStore,
    planning_llm: LLM,
    rewriting_llm: LLM,
    prompt_maker: PromptMaker,
    creative_latitude: str = "moderate",
    num_examples_per_paragraph: int = 3
) -> str:
    """
    Agent-based style rewriting WITHOUT craft annotations.

    Same as agent_rewrite(), but the rewriting prompt omits all craft metadata
    (function, craft_move, guidance, teaching_note, tags). Shows only:
    - Original paragraph text
    - Example passages (raw text only)

    This tests whether simpler prompts with less instructional scaffolding
    perform better than annotated prompts.

    Orchestrates 3-phase workflow:
    1. Planning: Analyze flattened text, create paragraph-level plans
    2. Retrieval: Search catalog for craft-tagged examples
    3. Rewriting: Generate styled output using ONLY example texts (no annotations)

    Args:
        flattened_content: Style-flattened input text to rewrite
        segment_store: SegmentStore instance with catalog of examples
        planning_llm: LLM instance for planning (recommended temp=0.5)
        rewriting_llm: LLM instance for rewriting (recommended temp=0.7)
        prompt_maker: PromptMaker instance for template rendering
        creative_latitude: "conservative", "moderate", or "aggressive"
        num_examples_per_paragraph: Number of examples to retrieve per paragraph

    Returns:
        str: Styled rewritten text (ONLY the text, no debugging output)

    Raises:
        ValueError: If planning fails schema validation or catalog search fails

    Example:
        >>> from belletrist import LLM, LLMConfig, PromptMaker, SegmentStore
        >>> from belletrist.agent_rewriter import agent_rewrite_no_annotations
        >>>
        >>> planning_llm = LLM(LLMConfig(model="gpt-4", api_key=key, temperature=0.5))
        >>> rewriting_llm = LLM(LLMConfig(model="gpt-4", api_key=key, temperature=0.7))
        >>>
        >>> with SegmentStore("segments.db") as store:
        >>>     styled_text = agent_rewrite_no_annotations(
        >>>         flattened_content="Democracy has flaws. But it remains best.",
        >>>         segment_store=store,
        >>>         planning_llm=planning_llm,
        >>>         rewriting_llm=rewriting_llm,
        >>>         prompt_maker=PromptMaker()
        >>>     )
        >>> print(styled_text)  # Clean output, no debugging
    """
    # Phase 1: Planning (silently) - SAME as annotated version
    available_tags = list(segment_store.list_all_tags().keys())
    plan = _plan_rewrite(
        flattened_text=flattened_content,
        available_tags=available_tags,
        llm=planning_llm,
        prompt_maker=prompt_maker,
        creative_latitude=creative_latitude
    )

    # Phase 2: Retrieval (silently) - SAME as annotated version
    examples = _retrieve_examples(
        plan=plan,
        store=segment_store,
        num_examples=num_examples_per_paragraph
    )

    # Phase 3: Rewriting (silently) - DIFFERENT: uses no-annotations template
    styled_text = _rewrite_with_style_no_annotations(
        plan=plan,
        examples=examples,
        llm=rewriting_llm,
        prompt_maker=prompt_maker
    )

    # Return ONLY the styled text, no debugging artifacts
    return styled_text


# =============================================================================
# Holistic Strategic Retrieval Workflow
# =============================================================================

def _plan_strategic_retrieval(
    flattened_text: str,
    available_tags: list[str],
    llm: LLM,
    prompt_maker: PromptMaker,
    creative_latitude: str = "moderate"
):
    """Phase 1: Analyze text holistically and select craft palette.

    Args:
        flattened_text: Complete flattened text (not per-paragraph)
        available_tags: All tags from segment catalog
        llm: Planning LLM (temp=0.5 for consistency)
        prompt_maker: PromptMaker instance
        creative_latitude: "conservative", "moderate", or "aggressive"

    Returns:
        StrategicRetrievalPlan with diagnosis and 6-12 selected tags

    Raises:
        ValueError: If schema validation fails
    """
    from belletrist.prompts import StrategicRetrievalPlannerConfig, StrategicRetrievalPlan
    from belletrist.prompts.canonical_tags import get_all_canonical_tags, format_for_jinja

    # Filter canonical tags → Tier 2 only
    canonical_set = get_all_canonical_tags()
    tier2_tags = [tag for tag in available_tags if tag not in canonical_set]

    # Format canonical tags for template
    canonical_tags_formatted = format_for_jinja()

    # Create config
    config = StrategicRetrievalPlannerConfig(
        flattened_text=flattened_text,
        tier2_tags=tier2_tags,
        canonical_tags_formatted=canonical_tags_formatted,
        creative_latitude=creative_latitude
    )

    # Render and call LLM
    prompt = prompt_maker.render(config)
    response = llm.complete_with_schema(
        prompt=prompt,
        schema_model=StrategicRetrievalPlan,
        system="You are a JSON API returning structured data. Always respond with valid JSON matching the schema."
    )

    return response.content  # Validated StrategicRetrievalPlan instance


def _retrieve_examples_holistic(
    plan,
    store: SegmentStore,
    target_count: int = 10
) -> list[dict]:
    """Phase 2: Retrieve strategic examples covering craft palette.

    Strategy:
    1. Collect all candidates from selected tags
    2. Deduplicate by segment_id
    3. Apply heuristics (length, tag specificity, source diversity)
    4. Return top N examples (8-10)

    Args:
        plan: StrategicRetrievalPlan with selected_tags
        store: SegmentStore instance
        target_count: Number of examples to return (8-10)

    Returns:
        List of dicts with: segment_id, craft_move, teaching_note, tags, text
    """
    # Collect candidates from all selected tags
    candidates = []
    for tag in plan.selected_tags:
        tag_results = store.search_by_tag(tag, exact_match=True)
        candidates.extend(tag_results)

    # Deduplicate by segment_id (preserving first occurrence)
    seen_ids = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate.segment_id not in seen_ids:
            unique_candidates.append(candidate)
            seen_ids.add(candidate.segment_id)

    # Apply selection heuristics (reuse existing function)
    selected = select_examples_with_heuristics(
        unique_candidates,
        max_examples=target_count
    )

    # Convert to dicts for template
    examples = [
        {
            'segment_id': seg.segment_id,
            'craft_move': seg.craft_move,
            'teaching_note': seg.teaching_note,
            'tags': seg.tags,
            'text': seg.text
        }
        for seg in selected
    ]

    return examples


def _rewrite_holistic(
    flattened_text: str,
    plan,
    examples: list[dict],
    llm: LLM,
    prompt_maker: PromptMaker,
    include_teaching_notes: bool = True
) -> str:
    """Phase 3: Generate styled output with holistic prompt.

    Args:
        flattened_text: Complete flattened text
        plan: StrategicRetrievalPlan with diagnosis
        examples: 8-10 retrieved examples
        llm: Rewriting LLM (temp=0.7 for creativity)
        prompt_maker: PromptMaker instance
        include_teaching_notes: Show teaching notes in prompt

    Returns:
        str: Styled rewritten text
    """
    from belletrist.prompts import HolisticStyledRewriteConfig

    # Create config
    config = HolisticStyledRewriteConfig(
        flattened_text=flattened_text,
        plan=plan,
        retrieved_examples=examples,
        include_teaching_notes=include_teaching_notes
    )

    # Render and call LLM (text output, not schema)
    prompt = prompt_maker.render(config)
    response = llm.complete(prompt)

    return response.content


def agent_rewrite_holistic(
    flattened_content: str,
    segment_store: SegmentStore,
    planning_llm: LLM,
    rewriting_llm: LLM,
    prompt_maker: PromptMaker,
    creative_latitude: str = "moderate",
    target_example_count: int = 10,
    include_teaching_notes: bool = True
) -> str:
    """
    Holistic strategic rewriting: analyzes entire text, selects craft palette,
    retrieves 8-10 strategic examples, passes to lean rewriter.

    This approach addresses the paragraph-level agent's poor performance (3.10 ranking)
    by using:
    - Holistic analysis (not per-paragraph)
    - 8-10 total examples (not 30+)
    - Lean rewriter template (closer to vanilla few-shot which ranks 1.30)

    Workflow:
    1. Plan: Analyze text holistically → select 6-12 strategic tags
    2. Retrieve: Collect candidates from tags → apply heuristics → return top 8-10
    3. Rewrite: Pass diagnosis + examples to simple rewriter

    Args:
        flattened_content: Style-flattened input text
        segment_store: SegmentStore with example catalog
        planning_llm: LLM for strategic planning (temp=0.5)
        rewriting_llm: LLM for rewriting (temp=0.7)
        prompt_maker: PromptMaker for templates
        creative_latitude: "conservative", "moderate", or "aggressive"
        target_example_count: Number of examples to retrieve (8-10)
        include_teaching_notes: Whether to show teaching notes in rewriter

    Returns:
        str: Styled rewritten text

    Example:
        >>> planning_llm = LLM(LLMConfig(model="gpt-4", temperature=0.5))
        >>> rewriting_llm = LLM(LLMConfig(model="gpt-4", temperature=0.7))
        >>> with SegmentStore("segments.db") as store:
        ...     styled = agent_rewrite_holistic(
        ...         flattened_content=content,
        ...         segment_store=store,
        ...         planning_llm=planning_llm,
        ...         rewriting_llm=rewriting_llm,
        ...         prompt_maker=PromptMaker(),
        ...         target_example_count=10
        ...     )
    """
    # Phase 1: Strategic planning (silently)
    available_tags = list(segment_store.list_all_tags().keys())
    plan = _plan_strategic_retrieval(
        flattened_text=flattened_content,
        available_tags=available_tags,
        llm=planning_llm,
        prompt_maker=prompt_maker,
        creative_latitude=creative_latitude
    )

    # Phase 2: Holistic retrieval (silently)
    examples = _retrieve_examples_holistic(
        plan=plan,
        store=segment_store,
        target_count=target_example_count
    )

    # Phase 3: Holistic rewriting (silently)
    styled_text = _rewrite_holistic(
        flattened_text=flattened_content,
        plan=plan,
        examples=examples,
        llm=rewriting_llm,
        prompt_maker=prompt_maker,
        include_teaching_notes=include_teaching_notes
    )

    # Return ONLY the styled text, no debugging artifacts
    return styled_text
