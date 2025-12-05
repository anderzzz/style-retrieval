"""
Style Retrieval: Segment catalog building and retrieval workflow.

This script demonstrates:
1. Loading a chapter via DataSampler
2. Analyzing with LLM to identify exemplary segments
3. Storing segments in SegmentStore
4. Retrieving segments via catalog browsing

Follows the Anthropic skills pattern: agents browse catalog, choose what to retrieve.
"""
import os
from pathlib import Path
from typing import List

from belletrist import LLM, LLMConfig, PromptMaker, DataSampler, SegmentStore
from belletrist.prompts import ExemplarySegmentAnalysisConfig, ExemplarySegmentAnalysis


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
    chapter_segment = sampler.get_paragraph_chunk(file_index, paragraph_range)

    print(f"\nAnalyzing: {chapter_segment.file_path.name}")
    print(f"Paragraphs: {chapter_segment.paragraph_start}-{chapter_segment.paragraph_end}")
    print(f"Length: {len(chapter_segment.text)} characters\n")

    # Configure prompt
    config = ExemplarySegmentAnalysisConfig(
        chapter_text=chapter_segment.text,
        file_name=chapter_segment.file_path.name
    )

    prompt = prompt_maker.render(config)

    # Call LLM with structured output
    print("Calling LLM for segment analysis...")
    response = llm.complete_with_schema(
        prompt=prompt,
        schema_model=ExemplarySegmentAnalysis,
        temperature=0.7  # Some creativity in selection
    )

    analysis: ExemplarySegmentAnalysis = response.content

    print(f"✓ Identified {len(analysis.segments)} exemplary segments")
    if analysis.analysis_notes:
        print(f"Notes: {analysis.analysis_notes[:200]}...")

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

    for seg in analysis.segments:
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
        print(f"  Saved: {segment_id} (paragraphs {abs_start}-{abs_end})")

    return segment_ids


def browse_and_retrieve_example(store: SegmentStore, sampler: DataSampler):
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
    print("\n1. Available tags in catalog:")
    tags = store.list_all_tags()
    for tag, count in list(tags.items())[:10]:  # Top 10
        print(f"   - {tag}: {count} segments")

    # Step 2: Browse catalog summaries
    print("\n2. Browsing catalog (first 5 segments):")
    catalog = store.browse_catalog(limit=5)
    for entry in catalog:
        print(f"\n   {entry['segment_id']} ({entry['file_name']})")
        print(f"   Range: paragraphs {entry['paragraph_range']}")
        print(f"   Function: {entry['functional_description'][:80]}...")
        print(f"   Form: {entry['formal_description'][:80]}...")
        print(f"   Tags: {', '.join(entry['tags'])}")

    # Step 3: Retrieve a specific segment
    if catalog:
        segment_id = catalog[0]['segment_id']
        print(f"\n3. Retrieving full segment: {segment_id}")
        record = store.get_segment(segment_id)

        if record:
            print(f"\n   Full text ({len(record.text)} chars):")
            print(f"   {record.text[:300]}...")

            # Demonstrate conversion back to TextSegment
            text_segment = record.to_text_segment(sampler)
            print(f"\n   Re-retrieved via DataSampler:")
            print(f"   File: {text_segment.file_path.name}")
            print(f"   Range: {text_segment.paragraph_start}-{text_segment.paragraph_end}")


def main():
    """Main workflow: analyze, store, browse, retrieve."""

    # Configuration
    DATA_PATH = Path(__file__).parent.parent / "data" / "russell"
    DB_PATH = Path(__file__).parent.parent / "segments.db"

    # Get API key (try multiple providers)
    api_key = (
        os.environ.get('ANTHROPIC_API_KEY') or
        os.environ.get('OPENAI_API_KEY') or
        os.environ.get('MISTRAL_API_KEY')
    )

    if not api_key:
        raise ValueError(
            "No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or MISTRAL_API_KEY"
        )

    # Determine model from available key
    if os.environ.get('ANTHROPIC_API_KEY'):
        model = "claude-3-5-sonnet-20241022"
    elif os.environ.get('OPENAI_API_KEY'):
        model = "gpt-4o"
    else:
        model = "mistral/mistral-large-2411"

    # Initialize components
    sampler = DataSampler(DATA_PATH)
    llm = LLM(LLMConfig(
        model=model,
        api_key=api_key,
        temperature=0.7
    ))
    prompt_maker = PromptMaker()

    print(f"Initialized with model: {model}")
    print(f"Data corpus: {len(sampler.fps)} files")

    # Create/open segment store
    with SegmentStore(DB_PATH) as store:

        # ANALYSIS PHASE: Analyze first chapter of first file
        # (In production, loop through all chapters/files)
        file_index = 0
        chapter_range = slice(0, 50)  # First ~50 paragraphs

        analysis = analyze_chapter(
            sampler=sampler,
            llm=llm,
            prompt_maker=prompt_maker,
            file_index=file_index,
            paragraph_range=chapter_range
        )

        # STORAGE PHASE: Store identified segments
        print("\nStoring segments in catalog...")
        segment_ids = store_segments(
            store=store,
            sampler=sampler,
            analysis=analysis,
            file_index=file_index,
            base_paragraph_offset=chapter_range.start
        )

        print(f"\n✓ Stored {len(segment_ids)} segments: {segment_ids[0]} ... {segment_ids[-1]}")

        # RETRIEVAL PHASE: Demonstrate browsing and retrieval
        browse_and_retrieve_example(store, sampler)

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)
    print(f"\nSegment catalog saved to: {DB_PATH}")
    print("Agents can now browse and retrieve examples via:")
    print("  - store.list_all_tags() → see available categories")
    print("  - store.browse_catalog() → read descriptions")
    print("  - store.get_segment(id) → retrieve full text")
    print("  - store.search_by_tag(tag) → filter by form/function")


if __name__ == "__main__":
    main()