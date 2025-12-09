"""
Chapter Preview Utility

Simple interactive tool to preview text segments from DataSampler.
Helps tune chapter boundaries before adding them to chapters_config.yaml.

USAGE:
    python runs/preview_chapter.py

Then follow the interactive prompts to preview different segments.
"""
import os
from pathlib import Path
from belletrist import DataSampler


def preview_segment(sampler: DataSampler, file_index: int, start: int, end: int):
    """Preview a text segment with useful statistics."""
    try:
        segment = sampler.get_paragraph_chunk(file_index, slice(start, end))

        print("\n" + "="*70)
        print("SEGMENT PREVIEW")
        print("="*70)
        print(f"File: {segment.file_path.name}")
        print(f"File index: {file_index}")
        print(f"Paragraph range: {start}-{end} ({end - start} paragraphs)")
        print(f"Characters: {len(segment.text):,}")
        print(f"Words (approx): {len(segment.text.split()):,}")
        print(f"Lines: {segment.text.count(chr(10)) + 1}")

        print("\n" + "-"*70)
        print("TEXT CONTENT:")
        print("-"*70)
        print(segment.text)
        print("-"*70)

        # Show first and last few words for quick reference
        words = segment.text.split()
        if len(words) > 20:
            print(f"\nFirst 10 words: {' '.join(words[:10])}...")
            print(f"Last 10 words: ...{' '.join(words[-10:])}")

        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")


def show_file_info(sampler: DataSampler):
    """Display available files and their paragraph counts."""
    print("\n" + "="*70)
    print("AVAILABLE FILES")
    print("="*70)
    for idx, fp in enumerate(sampler.fps):
        para_count = sampler.n_paragraphs[fp.name]
        print(f"  [{idx}] {fp.name} - {para_count:,} paragraphs")
    print("="*70)


def interactive_mode(sampler: DataSampler):
    """Interactive preview mode."""
    print("\n" + "="*70)
    print("CHAPTER PREVIEW UTILITY - Interactive Mode")
    print("="*70)
    print("Commands:")
    print("  'files' - Show available files")
    print("  'quit' or 'q' - Exit")
    print("  Or enter: file_index start end")
    print("="*70)

    while True:
        try:
            user_input = input("\n> ").strip().lower()

            if user_input in ['quit', 'q', 'exit']:
                print("\nGoodbye!")
                break

            if user_input == 'files':
                show_file_info(sampler)
                continue

            if user_input == 'help':
                print("\nEnter three numbers: file_index paragraph_start paragraph_end")
                print("Example: 0 9 41")
                print("Or type 'files' to see available files")
                continue

            # Parse input
            parts = user_input.split()
            if len(parts) != 3:
                print("✗ Please enter three numbers: file_index start end")
                print("  Example: 0 9 41")
                continue

            file_index = int(parts[0])
            start = int(parts[1])
            end = int(parts[2])

            # Validate
            if file_index < 0 or file_index >= len(sampler.fps):
                print(f"✗ Invalid file_index. Must be 0-{len(sampler.fps)-1}")
                print("  Type 'files' to see available files")
                continue

            if start < 0 or end <= start:
                print("✗ Invalid range. Start must be >= 0 and end must be > start")
                continue

            max_para = sampler.n_paragraphs[sampler.fps[file_index].name]
            if end > max_para:
                print(f"✗ End paragraph {end} exceeds file length ({max_para} paragraphs)")
                continue

            # Preview the segment
            preview_segment(sampler, file_index, start, end)

        except ValueError:
            print("✗ Invalid input. Enter three numbers: file_index start end")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    """Main entry point."""
    # Use same data path as main script
    data_path = Path(__file__).parent.parent / "data" / "russell"

    if not data_path.exists():
        print(f"✗ Data path not found: {data_path}")
        print("  Please ensure your data files are in the correct location.")
        return

    print("Loading data...")
    sampler = DataSampler(data_path)
    print(f"✓ Loaded {len(sampler.fps)} files")

    show_file_info(sampler)
    interactive_mode(sampler)


if __name__ == "__main__":
    main()
