"""
Style Rewrite: Multi-agent workflow for styled text generation.

This script demonstrates:
1. Planning Agent - analyzes flattened text, creates paragraph-level plan
2. Python Retrieval - searches catalog for matching craft examples (heuristic-based)
3. Rewriting Agent - uses examples to generate styled output

Follows Option B architecture: deterministic retrieval with LLM planning + rewriting.

USAGE:
    1. Set configuration parameters below (API key, model, paths, input text)
    2. Ensure segments.db exists (run style_retrieval.py first to build catalog)
    3. Run: python runs/style_rewrite.py
    4. Monitor output as workflow progresses
"""
import os
from pathlib import Path
from typing import List, Dict

from belletrist import LLM, LLMConfig, PromptMaker, DataSampler, SegmentStore
from belletrist.prompts import (
    StyleRewritePlannerConfig,
    StyledRewriteConfig,
    StyleRewritePlan,
    ParagraphPlan
)
from belletrist.segment_store import SegmentRecord


# ============================================================================
# CONFIGURATION - Modify these parameters before running
# ============================================================================

# API Configuration
API_KEY = os.environ.get('MISTRAL_API_KEY', '')
#API_KEY = os.environ.get('TOGETHER_AI_API_KEY', '')  # or set directly: "sk-..."
MODEL = "mistral/mistral-large-2512"
#MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507"

# Alternative examples:
# MODEL = "gpt-4o"
#
# API_KEY = os.environ.get('TOGETHER_AI_API_KEY', '')
# MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507"

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "russell"
DB_PATH = Path(__file__).parent.parent / "segments.db"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"

# LLM Parameters
PLANNING_TEMPERATURE = 0.5       # Lower for deterministic planning
REWRITING_TEMPERATURE = 0.7      # Moderate for creative rewriting

MAX_TOKENS_PLANNING = 4096
MAX_TOKENS_REWRITING = 8192

# Workflow Parameters
NUM_EXAMPLES_PER_PARAGRAPH = 3
TARGET_STYLE = "balanced, lucid, rhythmically varied with concessive structure"
CREATIVE_LATITUDE = "moderate"  # "conservative", "moderate", or "aggressive"
MAX_TAGS_TO_SHOW = 50  # Maximum tags to show in planning prompt (0 = show all)

# Input Text (flattened/style-sparse)
INPUT_TEXT = """
Democracy has flaws. It can be inefficient and messy. Leaders are elected by popularity rather than competence. The process is slow.

But democracy remains the best system. It provides accountability. Citizens can remove bad leaders. No other system has proven superior.

The alternatives are worse. Autocracy lacks checks on power. Technocracy ignores popular will. Aristocracy entrenches privilege.

Democracy's strength is its weakness. Slowness prevents rash action. Messiness reflects genuine debate. Inefficiency protects against tyranny.
"""

# ============================================================================


def plan_rewrite(
    flattened_text: str,
    available_tags: List[str],
    llm: LLM,
    prompt_maker: PromptMaker,
    creative_latitude: str = "moderate",
    max_tags_to_show: int = 50
) -> StyleRewritePlan:
    """Phase 1: Planning agent analyzes text and creates rewrite plan.

    Args:
        flattened_text: Style-sparse input text
        available_tags: All tags from segment catalog (will be filtered internally)
        llm: LLM instance
        prompt_maker: PromptMaker instance
        creative_latitude: "conservative", "moderate", or "aggressive"
        max_tags_to_show: Maximum tags to show in prompt (0 = show all; DEPRECATED - filtering now done internally)

    Returns:
        StyleRewritePlan with paragraph-level guidance
    """
    from belletrist.prompts.canonical_tags import get_all_canonical_tags, format_for_jinja

    print("\n[1/3] Planning Agent: Analyzing text structure...")

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
    print(f"      ✓ Prompt configured ({len(prompt):,} characters)")

    # Debug: Print prompt preview
    if os.environ.get('DEBUG_PROMPTS'):
        print(f"\n      Prompt preview (first 500 chars):")
        print(f"      {prompt[:500]}...")
        print()

    # Call LLM with schema
    print("      Calling LLM for structural analysis...")
    try:
        response = llm.complete_with_schema(
            prompt=prompt,
            schema_model=StyleRewritePlan,
            system="You are a JSON API that returns structured data. Always respond with valid JSON matching the requested schema. Never include explanatory text.",
            strict=False  # Some models (like Qwen) don't support strict schema mode
        )
    except ValueError as e:
        # If schema validation fails, try to get raw response for debugging
        print(f"\n      ✗ Schema validation failed!")
        print(f"      Error: {str(e)[:200]}...")
        print(f"\n      This might be a model-specific issue with {MODEL}")
        print(f"      Try using a different model (e.g., GPT-4, Claude) or check the prompt template.")
        raise

    plan: StyleRewritePlan = response.content

    # Debug: print validation mode
    print(f"      Schema validation mode: {response.schema_validation_mode}")

    print(f"      ✓ Plan complete!")
    print(f"      Identified {len(plan.paragraphs)} paragraphs")
    print(f"      Strategy: {plan.overall_strategy[:80]}...")

    return plan


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


def retrieve_examples(
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
    print("\n[2/3] Retrieval: Searching catalog for examples...")

    examples_by_paragraph = {}

    for para_plan in plan.paragraphs:
        para_id = para_plan.paragraph_id
        print(f"\n      Paragraph {para_id}: {para_plan.craft_move}")

        # Collect candidates from all tags
        candidates = []
        for tag in para_plan.craft_tags:
            tag_results = store.search_by_tag(tag, exact_match=True)
            candidates.extend(tag_results)
            print(f"        • {tag}: {len(tag_results)} matches")

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

        print(f"        ✓ Selected {len(selected)} examples")

    total_examples = sum(len(exs) for exs in examples_by_paragraph.values())
    print(f"\n      ✓ Retrieved {total_examples} total examples across {len(plan.paragraphs)} paragraphs")

    return examples_by_paragraph


def rewrite_with_style(
    plan: StyleRewritePlan,
    examples: Dict[int, List[dict]],
    llm: LLM,
    prompt_maker: PromptMaker
) -> str:
    """Phase 3: Rewriting agent generates styled output.

    Args:
        plan: StyleRewritePlan with paragraph guidance
        examples: Retrieved examples by paragraph_id
        llm: LLM instance
        prompt_maker: PromptMaker instance

    Returns:
        Styled rewritten text
    """
    print("\n[3/3] Rewriting Agent: Generating styled output...")

    # Create config
    config = StyledRewriteConfig(
        plan=plan,
        retrieved_examples=examples
    )

    # Render prompt
    prompt = prompt_maker.render(config)
    print(f"      ✓ Prompt configured ({len(prompt):,} characters)")
    print(f"      Paragraphs to rewrite: {len(plan.paragraphs)}")

    # Call LLM (text output, not schema)
    print("      Calling LLM for stylistic rewriting...")
    response = llm.complete(prompt)

    styled_text = response.content

    print(f"      ✓ Rewrite complete!")
    print(f"      Output length: {len(styled_text):,} characters")

    return styled_text


def main():
    """Main workflow: plan, retrieve, rewrite."""

    # Validate configuration
    print("="*60)
    print("STYLE REWRITE WORKFLOW")
    print("="*60)
    print("\n[Setup] Validating configuration...")

    if not API_KEY:
        raise ValueError(
            "API_KEY not set. Please configure API_KEY in the configuration section."
        )

    if not DB_PATH.exists():
        raise ValueError(
            f"Segment database not found: {DB_PATH}\n"
            f"Please run style_retrieval.py first to build the catalog."
        )

    print(f"        Model: {MODEL}")
    print(f"        Database: {DB_PATH}")
    print(f"        Target Style: {TARGET_STYLE}")
    print(f"        Creative Latitude: {CREATIVE_LATITUDE}")

    # Initialize components
    print("\n[Setup] Initializing components...")

    sampler = DataSampler(DATA_PATH)
    print(f"        ✓ DataSampler loaded {len(sampler.fps)} files")

    # Separate LLMs for planning and rewriting (different temperatures)
    planning_llm = LLM(LLMConfig(
        model=MODEL,
        api_key=API_KEY,
        temperature=PLANNING_TEMPERATURE,
        max_tokens=MAX_TOKENS_PLANNING
    ))
    print(f"        ✓ Planning LLM configured (temp={PLANNING_TEMPERATURE})")

    rewriting_llm = LLM(LLMConfig(
        model=MODEL,
        api_key=API_KEY,
        temperature=REWRITING_TEMPERATURE,
        max_tokens=MAX_TOKENS_REWRITING
    ))
    print(f"        ✓ Rewriting LLM configured (temp={REWRITING_TEMPERATURE})")

    prompt_maker = PromptMaker()
    print(f"        ✓ PromptMaker ready")

    # Open segment store
    print(f"\n[Setup] Opening segment database: {DB_PATH}")
    with SegmentStore(DB_PATH) as store:
        segment_count = store.get_count()
        print(f"        ✓ SegmentStore connected ({segment_count} segments)")

        # Get available tags
        all_tags = store.list_all_tags()
        available_tags = list(all_tags.keys())
        print(f"        ✓ Loaded {len(available_tags)} unique tags")

        # ====================================================================
        # PHASE 1: PLANNING
        # ====================================================================
        print("\n" + "="*60)
        print("PHASE 1: ANALYZE TEXT & CREATE REWRITE PLAN")
        print("="*60)

        print(f"\nInput text length: {len(INPUT_TEXT)} characters")
        print(f"Input preview: {INPUT_TEXT[:150]}...\n")

        plan = plan_rewrite(
            flattened_text=INPUT_TEXT,
            available_tags=available_tags,
            llm=planning_llm,
            prompt_maker=prompt_maker,
            creative_latitude=CREATIVE_LATITUDE,
            max_tags_to_show=MAX_TAGS_TO_SHOW
        )

        # Display plan summary
        print(f"\n✓ Plan Summary:")
        print(f"  Strategy: {plan.overall_strategy}")
        print(f"  Paragraphs: {len(plan.paragraphs)}")
        for para in plan.paragraphs:
            print(f"    [{para.paragraph_id}] {para.craft_move} → {para.craft_tags}")

        # ====================================================================
        # PHASE 2: RETRIEVAL
        # ====================================================================
        print("\n" + "="*60)
        print("PHASE 2: RETRIEVE EXEMPLARY PASSAGES")
        print("="*60)

        examples = retrieve_examples(
            plan=plan,
            store=store,
            num_examples=NUM_EXAMPLES_PER_PARAGRAPH
        )

        # ====================================================================
        # PHASE 3: REWRITING
        # ====================================================================
        print("\n" + "="*60)
        print("PHASE 3: GENERATE STYLED OUTPUT")
        print("="*60)

        styled_text = rewrite_with_style(
            plan=plan,
            examples=examples,
            llm=rewriting_llm,
            prompt_maker=prompt_maker
        )

        # ====================================================================
        # OUTPUT
        # ====================================================================
        print("\n" + "="*60)
        print("WORKFLOW COMPLETE")
        print("="*60)

        print("\n" + "─"*60)
        print("ORIGINAL TEXT:")
        print("─"*60)
        print(INPUT_TEXT)

        print("\n" + "─"*60)
        print("STYLED OUTPUT:")
        print("─"*60)
        print(styled_text)
        print("─"*60)

        # Save output
        OUTPUT_PATH.mkdir(exist_ok=True)
        output_file = OUTPUT_PATH / "rewrite_output.txt"
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ORIGINAL TEXT\n")
            f.write("="*60 + "\n\n")
            f.write(INPUT_TEXT)
            f.write("\n\n")
            f.write("="*60 + "\n")
            f.write("STYLED OUTPUT\n")
            f.write("="*60 + "\n\n")
            f.write(styled_text)

        print(f"\n✓ Output saved to: {output_file}")
        print("="*60)


if __name__ == "__main__":
    main()
