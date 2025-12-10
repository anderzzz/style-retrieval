"""
Style Retrieval: Segment catalog building and retrieval workflow.

This script demonstrates:
1. Loading chapters via DataSampler
2. Analyzing with LLM to identify exemplary segments
3. Storing segments in SegmentStore
4. Retrieving segments via catalog browsing

Follows the Anthropic skills pattern: agents browse catalog, choose what to retrieve.

USAGE:
    1. Set configuration parameters below (API key, model, data paths, etc.)
    2. Define chapters in chapters_config.yaml
    3. Run: python runs/style_retrieval.py
    4. Monitor output as workflow progresses
"""
import os
import yaml
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from belletrist import LLM, LLMConfig, PromptMaker, DataSampler, SegmentStore
from belletrist.prompts import ExemplarySegmentAnalysisConfig, ExemplarySegmentAnalysis
from belletrist.prompts.canonical_tags import get_all_canonical_tags, format_for_jinja


# ============================================================================
# CHAPTER CONFIGURATION MODEL
# ============================================================================

class ChapterConfig(BaseModel):
    """Configuration for a single chapter to analyze."""
    file_index: int = Field(..., ge=0, description="Index of file in data directory")
    paragraph_start: int = Field(..., ge=0, description="Starting paragraph (inclusive)")
    paragraph_end: int = Field(..., gt=0, description="Ending paragraph (exclusive)")
    description: str = Field(..., min_length=1, description="Human-readable chapter description")
    enabled: bool = Field(default=True, description="Whether to process this chapter")

    @property
    def paragraph_range(self) -> slice:
        """Convert to slice for DataSampler."""
        return slice(self.paragraph_start, self.paragraph_end)


class ChaptersConfig(BaseModel):
    """Root configuration containing all chapters."""
    chapters: List[ChapterConfig] = Field(..., min_items=1)


# ============================================================================
# CONFIGURATION - Modify these parameters before running
# ============================================================================

# API Configuration
# Set your API key and corresponding model
API_KEY = os.environ.get('TOGETHER_AI_API_KEY', '')  # or set directly: "sk-..."
#MODEL = "mistral/mistral-large-2411"
MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507"

# Alternative examples:
# API_KEY = os.environ.get('OPENAI_API_KEY', '')
# MODEL = "gpt-4o"
#
# API_KEY = os.environ.get('MISTRAL_API_KEY', '')
# MODEL = "mistral/mistral-large-2411"

# Data Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "russell"
DB_PATH = Path(__file__).parent.parent / "segments.db"
CHAPTERS_CONFIG_PATH = Path(__file__).parent.parent / "chapters_config.yaml"

# Analysis Parameters
TEMPERATURE = 0.7  # LLM temperature for segment selection

# Processing Control
SKIP_EXISTING_CHAPTERS = True  # Skip chapters that have already been processed
                               # Set to False to reprocess all chapters

# Catalog Browsing
CATALOG_PREVIEW_LIMIT = 5  # Number of segments to show in browse demo
TAG_PREVIEW_LIMIT = 10  # Number of tags to show in tag list

# ============================================================================


def load_chapters_config(config_path: Path) -> ChaptersConfig:
    """Load and validate chapters configuration from YAML file.

    Args:
        config_path: Path to chapters_config.yaml

    Returns:
        Validated ChaptersConfig with list of chapters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML is invalid or validation fails
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Chapters configuration not found: {config_path}\n"
            f"Please create chapters_config.yaml with chapter definitions."
        )

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    try:
        config = ChaptersConfig(**data)
    except Exception as e:
        raise ValueError(f"Invalid chapters configuration: {e}")

    return config


def chapter_already_processed(
    store: SegmentStore,
    chapter: ChapterConfig
) -> bool:
    """Check if a chapter has already been processed.

    Args:
        store: SegmentStore to query
        chapter: ChapterConfig to check

    Returns:
        True if segments exist for this chapter's paragraph range
    """
    # Query database for segments matching this file and paragraph range
    cursor = store.conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM segments
        WHERE file_index = ?
        AND paragraph_start >= ?
        AND paragraph_end <= ?
        """,
        (chapter.file_index, chapter.paragraph_start, chapter.paragraph_end)
    )
    count = cursor.fetchone()[0]
    return count > 0


def analyze_chapter(
    sampler: DataSampler,
    llm: LLM,
    prompt_maker: PromptMaker,
    file_index: int,
    paragraph_range: slice,
    existing_tier2_tags: List[str] = None,
    canonical_tags_formatted: List[dict] = None
) -> ExemplarySegmentAnalysis:
    """Analyze a chapter to identify exemplary segments.

    Args:
        sampler: DataSampler for loading text
        llm: LLM for analysis
        prompt_maker: PromptMaker for template rendering
        file_index: File to analyze
        paragraph_range: Chapter range (e.g., slice(0, 50))
        existing_tier2_tags: Author-specific tags already in catalog (Tier 2 only)
        canonical_tags_formatted: Formatted canonical tags for template injection

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
        file_name=chapter_segment.file_path.name,
        num_segments=5,
        existing_tier2_tags=existing_tier2_tags or [],
        canonical_tags_formatted=canonical_tags_formatted or []
    )

    if existing_tier2_tags:
        print(f"      Encouraging reuse of {len(existing_tier2_tags)} existing Tier 2 tags")
    prompt = prompt_maker.render(config)
    print(f"      ✓ Prompt configured ({len(prompt):,} characters)")

    # Call LLM with structured output
    print("\n[3/3] Calling LLM for segment analysis...")
    response = llm.complete_with_schema(
        prompt=prompt,
        schema_model=ExemplarySegmentAnalysis,
    )

    analysis: ExemplarySegmentAnalysis = response.content

    print(f"\n      ✓ Analysis complete!")
    print(f"      Identified {len(analysis.passages)} exemplary passages")
    if analysis.overall_observations:
        print(f"      Observations: {analysis.overall_observations[:150]}...")

    return analysis


def find_passage_in_chapter(
    passage_text: str,
    chapter_text: str,
    sampler: DataSampler,
    file_index: int,
    chapter_start_paragraph: int
) -> tuple[int, int] | None:
    """Find paragraph range for a passage within a chapter.

    Args:
        passage_text: The extracted passage text to locate
        chapter_text: Full chapter text
        sampler: DataSampler for accessing paragraphs
        file_index: File index
        chapter_start_paragraph: Starting paragraph index of the chapter

    Returns:
        Tuple of (paragraph_start, paragraph_end) or None if not found
    """
    # Normalize text for comparison (remove extra whitespace)
    normalized_passage = ' '.join(passage_text.split())

    # Try to find the passage in the chapter text
    if normalized_passage not in ' '.join(chapter_text.split()):
        return None

    # Iterate through paragraphs to find the match
    # Start with single paragraphs, then expand to multi-paragraph chunks
    file_path = sampler.fps[file_index]
    max_paragraphs = sampler.n_paragraphs[file_path.name]

    for length in range(1, 10):  # Try up to 10 paragraphs
        for start_offset in range(0, 50):  # Search within first 50 paragraphs
            abs_start = chapter_start_paragraph + start_offset
            abs_end = abs_start + length

            # Don't go beyond file bounds
            if abs_end > max_paragraphs:
                break

            chunk = sampler.get_paragraph_chunk(file_index, slice(abs_start, abs_end))
            normalized_chunk = ' '.join(chunk.text.split())

            if normalized_passage in normalized_chunk:
                return (abs_start, abs_end)

    return None


def store_segments(
    store: SegmentStore,
    sampler: DataSampler,
    analysis: ExemplarySegmentAnalysis,
    file_index: int,
    chapter_text: str,
    base_paragraph_offset: int = 0
) -> List[str]:
    """Store identified passages in the catalog.

    Args:
        store: SegmentStore for saving
        sampler: DataSampler for paragraph lookup
        analysis: Analysis results with passages
        file_index: File index being processed
        chapter_text: Full chapter text for locating passages
        base_paragraph_offset: Starting paragraph index of the chapter

    Returns:
        List of generated segment_ids
    """
    segment_ids = []

    for i, passage in enumerate(analysis.passages, 1):
        print(f"\n  [{i}/{len(analysis.passages)}] Processing: {passage.craft_move}")

        # Try to find passage location in chapter
        para_range = find_passage_in_chapter(
            passage.text,
            chapter_text,
            sampler,
            file_index,
            base_paragraph_offset
        )

        if para_range is None:
            print(f"       ⚠ Warning: Could not locate passage in chapter, using approximate range")
            # Use approximate range based on position
            para_start = base_paragraph_offset
            para_end = base_paragraph_offset + 1
        else:
            para_start, para_end = para_range

        # Retrieve TextSegment from sampler using found paragraph range
        text_segment = sampler.get_paragraph_chunk(file_index, slice(para_start, para_end))

        # Save to store
        segment_id = store.save_segment(
            text_segment=text_segment,
            craft_move=passage.craft_move,
            teaching_note=passage.teaching_note,
            tags=passage.tags
        )

        segment_ids.append(segment_id)
        print(f"       ✓ Saved: {segment_id}")
        print(f"       Paragraphs: {para_start}-{para_end}")
        print(f"       Tags: {', '.join(passage.tags)}")

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
        print(f"      Craft Move: {entry['craft_move']}")
        print(f"      Teaching Note: {entry['teaching_note'][:80]}...")
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
    print("STYLE RETRIEVAL WORKFLOW - MULTI-CHAPTER")
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

    # Load chapters configuration
    print(f"\n[Setup] Loading chapters configuration from: {CHAPTERS_CONFIG_PATH}")
    chapters_config = load_chapters_config(CHAPTERS_CONFIG_PATH)

    enabled_chapters = [ch for ch in chapters_config.chapters if ch.enabled]
    total_chapters = len(enabled_chapters)

    print(f"        ✓ Loaded {total_chapters} enabled chapters")
    print(f"        (Skipping {len(chapters_config.chapters) - total_chapters} disabled chapters)")

    print(f"\n        Model: {MODEL}")
    print(f"        Data: {DATA_PATH}")
    print(f"        Database: {DB_PATH}")
    print(f"        Chapters to process: {total_chapters}")

    # Initialize components
    print("\n[Setup] Initializing components...")
    sampler = DataSampler(DATA_PATH)
    print(f"        ✓ DataSampler loaded {len(sampler.fps)} files")

    llm = LLM(LLMConfig(
        model=MODEL,
        api_key=API_KEY,
        temperature=TEMPERATURE,
        max_tokens=16384  # Ensure enough tokens for full JSON response with 12+ passages
    ))
    print(f"        ✓ LLM configured: {MODEL} (temp={TEMPERATURE}, max_tokens=16384)")

    prompt_maker = PromptMaker()
    print(f"        ✓ PromptMaker ready")

    # Open/create segment store
    print(f"\n[Setup] Opening segment database: {DB_PATH}")
    with SegmentStore(DB_PATH) as store:
        print(f"        ✓ SegmentStore connected")

        # Track overall progress
        all_segment_ids = []
        processed_count = 0
        skipped_count = 0
        failed_chapters = []

        # ====================================================================
        # PHASE 1 & 2: LOOP OVER CHAPTERS - ANALYZE AND STORE
        # ====================================================================
        print("\n" + "="*60)
        print("PROCESSING CHAPTERS")
        print("="*60)
        if SKIP_EXISTING_CHAPTERS:
            print("Mode: Skip already-processed chapters")
        else:
            print("Mode: Reprocess all chapters")

        for chapter_idx, chapter in enumerate(enabled_chapters, 1):
            print(f"\n{'='*60}")
            print(f"CHAPTER {chapter_idx}/{total_chapters}: {chapter.description}")
            print(f"{'='*60}")
            print(f"File: {chapter.file_index}, Paragraphs: {chapter.paragraph_start}-{chapter.paragraph_end}")

            # Check if chapter already processed (if enabled)
            if SKIP_EXISTING_CHAPTERS and chapter_already_processed(store, chapter):
                print(f"\n⏭ Skipping - chapter already processed (SKIP_EXISTING_CHAPTERS=True)")
                skipped_count += 1
                continue

            try:
                # Get existing tags and filter to Tier 2 only (updated each iteration)
                existing_tags_dict = store.list_all_tags()
                all_tags = list(existing_tags_dict.keys()) if existing_tags_dict else []

                # Filter out canonical tags to get only Tier 2
                canonical_tag_set = get_all_canonical_tags()
                existing_tier2_tags = [tag for tag in all_tags if tag not in canonical_tag_set]

                # Get formatted canonical tags for template injection
                canonical_tags_formatted = format_for_jinja()

                if chapter_idx == 1:
                    if existing_tier2_tags:
                        print(f"\nCatalog contains {len(existing_tier2_tags)} author-specific tags (Tier 2)")
                        print(f"Will encourage reuse for consistency")
                    else:
                        print("\nNo Tier 2 tags yet - LLM will create author-specific vocabulary")
                else:
                    print(f"\nCatalog now contains {len(existing_tier2_tags)} author-specific tags (Tier 2)")

                # ANALYSIS
                print("\n[PHASE 1] Analyzing chapter for exemplary segments...")
                analysis = analyze_chapter(
                    sampler=sampler,
                    llm=llm,
                    prompt_maker=prompt_maker,
                    file_index=chapter.file_index,
                    paragraph_range=chapter.paragraph_range,
                    existing_tier2_tags=existing_tier2_tags,
                    canonical_tags_formatted=canonical_tags_formatted
                )

                # STORAGE
                print("\n[PHASE 2] Storing passages in catalog...")
                print(f"Storing {len(analysis.passages)} identified passages...")

                # Get chapter text for passage location
                chapter_segment = sampler.get_paragraph_chunk(
                    chapter.file_index,
                    chapter.paragraph_range
                )

                segment_ids = store_segments(
                    store=store,
                    sampler=sampler,
                    analysis=analysis,
                    file_index=chapter.file_index,
                    chapter_text=chapter_segment.text,
                    base_paragraph_offset=chapter.paragraph_start
                )

                all_segment_ids.extend(segment_ids)
                processed_count += 1

                print(f"\n✓ Chapter {chapter_idx} complete: {len(segment_ids)} passages stored")
                print(f"  Progress: {processed_count}/{total_chapters} chapters processed")

            except Exception as e:
                print(f"\n✗ ERROR processing chapter {chapter_idx}: {e}")
                failed_chapters.append((chapter_idx, chapter.description, str(e)))
                print(f"  Continuing with next chapter...")
                continue

        # Summary of processing
        print(f"\n{'='*60}")
        print(f"CHAPTER PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {processed_count}/{total_chapters} chapters")
        if skipped_count > 0:
            print(f"Skipped (already processed): {skipped_count} chapters")
        print(f"Total segments stored (this run): {len(all_segment_ids)}")

        if failed_chapters:
            print(f"\nFailed chapters ({len(failed_chapters)}):")
            for idx, desc, error in failed_chapters:
                print(f"  - Chapter {idx} ({desc}): {error}")
        elif processed_count > 0:
            print("\n✓ All processed chapters completed successfully!")

        if skipped_count == total_chapters:
            print("\n⏭ All chapters were already processed - no new segments added")
            print("  Set SKIP_EXISTING_CHAPTERS=False to reprocess")

        # ====================================================================
        # PHASE 3: RETRIEVAL DEMO
        # ====================================================================
        if all_segment_ids:
            print("\n" + "="*60)
            print("PHASE 3: DEMONSTRATE CATALOG RETRIEVAL")
            print("="*60)

            browse_and_retrieve_example(
                store=store,
                sampler=sampler,
                catalog_limit=CATALOG_PREVIEW_LIMIT,
                tag_limit=TAG_PREVIEW_LIMIT
            )
        else:
            print("\n⚠ No segments stored - skipping retrieval demo")

    # ====================================================================
    # COMPLETION
    # ====================================================================
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nSegment catalog saved to: {DB_PATH}")
    print(f"\nThis run:")
    print(f"  Chapters processed: {processed_count}/{total_chapters}")
    if skipped_count > 0:
        print(f"  Chapters skipped: {skipped_count}")
    print(f"  New segments stored: {len(all_segment_ids)}")

    if all_segment_ids:
        print(f"    First: {all_segment_ids[0]}")
        print(f"    Last:  {all_segment_ids[-1]}")

    # Show total catalog size
    with SegmentStore(DB_PATH) as store:
        total_in_catalog = store.get_count()
        total_tags = len(store.list_all_tags())

    print(f"\nCatalog totals:")
    print(f"  Total segments: {total_in_catalog}")
    print(f"  Unique tags: {total_tags}")

    print("\nNext steps - agents can now:")
    print("  • store.list_all_tags() → discover available categories")
    print("  • store.browse_catalog() → read segment descriptions")
    print("  • store.get_segment(id) → retrieve full text")
    print("  • store.search_by_tag(tag) → filter by form/function")
    print("="*60)


if __name__ == "__main__":
    main()