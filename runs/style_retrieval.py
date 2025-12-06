"""
Style Retrieval: Segment catalog building and retrieval workflow.

This script demonstrates:
1. Loading a chapter via DataSampler
2. Analyzing with LLM to identify exemplary segments
3. Storing segments in SegmentStore
4. Retrieving segments via catalog browsing

Follows the Anthropic skills pattern: agents browse catalog, choose what to retrieve.

USAGE:
    1. Set configuration parameters below (API key, model, data paths, etc.)
    2. Run: python runs/style_retrieval.py
    3. Monitor output as workflow progresses
"""
import os
from pathlib import Path
from typing import List

from belletrist import LLM, LLMConfig, PromptMaker, DataSampler, SegmentStore
from belletrist.prompts import ExemplarySegmentAnalysisConfig, ExemplarySegmentAnalysis


# ============================================================================
# CONFIGURATION - Modify these parameters before running
# ============================================================================

# API Configuration
# Set your API key and corresponding model
API_KEY = os.environ.get('MISTRAL_API_KEY', '')  # or set directly: "sk-..."
MODEL = "mistral/mistral-large-2411"

# Alternative examples:
# API_KEY = os.environ.get('OPENAI_API_KEY', '')
# MODEL = "gpt-4o"
#
# API_KEY = os.environ.get('MISTRAL_API_KEY', '')
# MODEL = "mistral/mistral-large-2411"

# Data Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "russell"
DB_PATH = Path(__file__).parent.parent / "segments.db"

# Analysis Parameters
FILE_INDEX = 0  # Which file to analyze (0 = first file)
CHAPTER_START = 9  # Starting paragraph
CHAPTER_END = 41  # Ending paragraph (first ~50 paragraphs)
TEMPERATURE = 0.7  # LLM temperature for segment selection

# Catalog Browsing
CATALOG_PREVIEW_LIMIT = 5  # Number of segments to show in browse demo
TAG_PREVIEW_LIMIT = 10  # Number of tags to show in tag list

# ============================================================================


def analyze_chapter(
    sampler: DataSampler,
    llm: LLM,
    prompt_maker: PromptMaker,
    file_index: int,
    paragraph_range: slice
) -> ExemplarySegmentAnalysis:
    """Analyze a chapter to identify exemplary segments.

    Args:
        sampler: DataSampler for loading text
        llm: LLM for analysis
        prompt_maker: PromptMaker for template rendering
        file_index: File to analyze
        paragraph_range: Chapter range (e.g., slice(0, 50))

    Returns:
        ExemplarySegmentAnalysis with identified segments
    """
    # Load chapter
    print("\n[1/3] Loading chapter text...")
    chapter_segment = sampler.get_paragraph_chunk(file_index, paragraph_range)

    print(f"      File: {chapter_segment.file_path.name}")
    print(f"      Paragraphs: {chapter_segment.paragraph_start}-{chapter_segment.paragraph_end}")
    print(f"      Length: {len(chapter_segment.text):,} characters")

    # Configure prompt
    print("\n[2/3] Preparing analysis prompt...")
    config = ExemplarySegmentAnalysisConfig(
        chapter_text=chapter_segment.text,
        file_name=chapter_segment.file_path.name
    )
    prompt = prompt_maker.render(config)
    print(f"      ✓ Prompt configured ({len(prompt):,} characters)")

    # Call LLM with structured output
    print("\n[3/3] Calling LLM for segment analysis...")
    response = llm.complete_with_schema(
        prompt=prompt,
        schema_model=ExemplarySegmentAnalysis,
        temperature=0.7  # Some creativity in selection
    )

    analysis: ExemplarySegmentAnalysis = response.content

    print(f"\n      ✓ Analysis complete!")
    print(f"      Identified {len(analysis.segments)} exemplary segments")
    if analysis.analysis_notes:
        print(f"      Notes: {analysis.analysis_notes[:150]}...")

    return analysis


def store_segments(
    store: SegmentStore,
    sampler: DataSampler,
    analysis: ExemplarySegmentAnalysis,
    file_index: int,
    base_paragraph_offset: int = 0
) -> List[str]:
    """Store identified segments in the catalog.

    Args:
        store: SegmentStore for saving
        sampler: DataSampler for retrieving exact text
        analysis: Analysis results with segments
        file_index: File index being processed
        base_paragraph_offset: Offset to add to paragraph indices (if analyzing sub-chapter)

    Returns:
        List of generated segment_ids
    """
    segment_ids = []

    for i, seg in enumerate(analysis.segments, 1):
        # Calculate absolute paragraph range
        abs_start = base_paragraph_offset + seg.paragraph_start
        abs_end = base_paragraph_offset + seg.paragraph_end

        # Retrieve exact text via DataSampler
        text_segment = sampler.get_paragraph_chunk(
            file_index,
            slice(abs_start, abs_end)
        )

        # Save to store
        segment_id = store.save_segment(
            text_segment=text_segment,
            functional_description=seg.functional_description,
            formal_description=seg.formal_description,
            tags=seg.suggested_tags
        )

        segment_ids.append(segment_id)
        print(f"  [{i}/{len(analysis.segments)}] {segment_id}")
        print(f"       Paragraphs: {abs_start}-{abs_end}")
        print(f"       Tags: {', '.join(seg.suggested_tags)}")

    return segment_ids


def browse_and_retrieve_example(
    store: SegmentStore,
    sampler: DataSampler,
    catalog_limit: int = 5,
    tag_limit: int = 10
):
    """Demonstrate catalog browsing and retrieval (skills pattern).

    This simulates how an agent would:
    1. List available tags
    2. Browse catalog summaries
    3. Retrieve specific segments
    """
    print("\n" + "="*60)
    print("CATALOG BROWSING DEMONSTRATION (Skills Pattern)")
    print("="*60)

    # Step 1: List available tags
    print("\n[1/3] Listing available tags...")
    tags = store.list_all_tags()
    print(f"      Found {len(tags)} unique tags in catalog")
    print(f"\n      Top {tag_limit} tags:")
    for tag, count in list(tags.items())[:tag_limit]:
        print(f"      - {tag}: {count} segments")

    # Step 2: Browse catalog summaries
    print(f"\n[2/3] Browsing catalog (showing {catalog_limit} segments)...")
    catalog = store.browse_catalog(limit=catalog_limit)
    print(f"      Retrieved {len(catalog)} segment summaries")
    for i, entry in enumerate(catalog, 1):
        print(f"\n      Segment {i}/{len(catalog)}: {entry['segment_id']}")
        print(f"      File: {entry['file_name']}")
        print(f"      Range: paragraphs {entry['paragraph_range']}")
        print(f"      Function: {entry['functional_description'][:80]}...")
        print(f"      Form: {entry['formal_description'][:80]}...")
        print(f"      Tags: {', '.join(entry['tags'])}")

    # Step 3: Retrieve a specific segment
    if catalog:
        segment_id = catalog[0]['segment_id']
        print(f"\n[3/3] Retrieving full text for segment: {segment_id}")
        record = store.get_segment(segment_id)

        if record:
            print(f"      ✓ Retrieved {len(record.text)} characters")
            print(f"\n      Preview (first 300 chars):")
            print(f"      {record.text[:300]}...")

            # Demonstrate conversion back to TextSegment
            text_segment = record.to_text_segment(sampler)
            print(f"\n      ✓ Re-retrieved via DataSampler:")
            print(f"        File: {text_segment.file_path.name}")
            print(f"        Range: {text_segment.paragraph_start}-{text_segment.paragraph_end}")


def main():
    """Main workflow: analyze, store, browse, retrieve."""

    # Validate configuration
    print("="*60)
    print("STYLE RETRIEVAL WORKFLOW")
    print("="*60)
    print("\n[Setup] Validating configuration...")

    if not API_KEY:
        raise ValueError(
            "API_KEY not set. Please configure API_KEY in the configuration section."
        )

    if not DATA_PATH.exists():
        raise ValueError(
            f"DATA_PATH does not exist: {DATA_PATH}\n"
            f"Please ensure your data files are in the correct location."
        )

    print(f"        Model: {MODEL}")
    print(f"        Data: {DATA_PATH}")
    print(f"        Database: {DB_PATH}")
    print(f"        Target: File {FILE_INDEX}, paragraphs {CHAPTER_START}-{CHAPTER_END}")

    # Initialize components
    print("\n[Setup] Initializing components...")
    sampler = DataSampler(DATA_PATH)
    print(f"        ✓ DataSampler loaded {len(sampler.fps)} files")

    llm = LLM(LLMConfig(
        model=MODEL,
        api_key=API_KEY,
        temperature=TEMPERATURE
    ))
    print(f"        ✓ LLM configured: {MODEL} (temp={TEMPERATURE})")

    prompt_maker = PromptMaker()
    print(f"        ✓ PromptMaker ready")

    # Open/create segment store
    print(f"\n[Setup] Opening segment database: {DB_PATH}")
    with SegmentStore(DB_PATH) as store:
        print(f"        ✓ SegmentStore connected")

        # ====================================================================
        # PHASE 1: ANALYSIS
        # ====================================================================
        print("\n" + "="*60)
        print("PHASE 1: ANALYZE CHAPTER FOR EXEMPLARY SEGMENTS")
        print("="*60)

        chapter_range = slice(CHAPTER_START, CHAPTER_END)

        analysis = analyze_chapter(
            sampler=sampler,
            llm=llm,
            prompt_maker=prompt_maker,
            file_index=FILE_INDEX,
            paragraph_range=chapter_range
        )

        # ====================================================================
        # PHASE 2: STORAGE
        # ====================================================================
        print("\n" + "="*60)
        print("PHASE 2: STORE SEGMENTS IN CATALOG")
        print("="*60)
        print(f"\nStoring {len(analysis.segments)} identified segments...")

        segment_ids = store_segments(
            store=store,
            sampler=sampler,
            analysis=analysis,
            file_index=FILE_INDEX,
            base_paragraph_offset=chapter_range.start
        )

        print(f"\n✓ Successfully stored {len(segment_ids)} segments")
        print(f"  First: {segment_ids[0]}")
        print(f"  Last:  {segment_ids[-1]}")

        # ====================================================================
        # PHASE 3: RETRIEVAL DEMO
        # ====================================================================
        print("\n" + "="*60)
        print("PHASE 3: DEMONSTRATE CATALOG RETRIEVAL")
        print("="*60)

        browse_and_retrieve_example(
            store=store,
            sampler=sampler,
            catalog_limit=CATALOG_PREVIEW_LIMIT,
            tag_limit=TAG_PREVIEW_LIMIT
        )

    # ====================================================================
    # COMPLETION
    # ====================================================================
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nSegment catalog saved to: {DB_PATH}")
    print(f"Total segments stored: {len(segment_ids)}")
    print("\nNext steps - agents can now:")
    print("  • store.list_all_tags() → discover available categories")
    print("  • store.browse_catalog() → read segment descriptions")
    print("  • store.get_segment(id) → retrieve full text")
    print("  • store.search_by_tag(tag) → filter by form/function")
    print("="*60)


if __name__ == "__main__":
    main()