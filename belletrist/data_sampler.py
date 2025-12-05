"""
Data sampler for loading and sampling text paragraphs from files.
"""
from typing import List, Optional, Union, Iterator
from pathlib import Path
from dataclasses import dataclass
import random


@dataclass
class TextSegment:
    """A segment of text with complete provenance information.

    This unified return type ensures all DataSampler methods provide
    consistent metadata for storage and analysis.
    """
    paragraphs: List[str]
    file_index: int
    paragraph_start: int
    paragraph_end: int
    file_path: Optional[Path] = None

    @property
    def text(self) -> str:
        """Join paragraphs with double newlines."""
        return "\n\n".join(self.paragraphs)

    @property
    def paragraph_count(self) -> int:
        """Number of paragraphs in this segment."""
        return len(self.paragraphs)

    def as_dict(self) -> dict:
        """Convert to dict for ResultStore compatibility."""
        return {
            'text': self.text,
            'file_index': self.file_index,
            'paragraph_start': self.paragraph_start,
            'paragraph_end': self.paragraph_end
        }


class ParagraphIndexError(Exception):
    """Raised when paragraph indexing is invalid."""
    pass


def load_paragraphs(
    file_path: Path,
    paragraph_range: Optional[Union[slice, int]] = None
) -> List[str]:
    """Load paragraphs from a file, optionally selecting a range.

    Paragraphs are split on double newlines.

    Args:
        file_path: Path to the text file.
        paragraph_range: Slice (e.g., 3:10), int (e.g., 5), or None for all.

    Returns:
        List of paragraph strings.
    """
    text = file_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if paragraph_range is None:
        return paragraphs
    elif isinstance(paragraph_range, int):
        return [paragraphs[paragraph_range]]
    elif isinstance(paragraph_range, slice):
        return paragraphs[paragraph_range]
    else:
        raise ValueError("paragraph_range must be None, int, or slice")


class DataSampler:
    """Sample paragraphs from text files in the data directory.

    Args:
        data_path: Path to the data directory.

    """
    def __init__(self, data_path: Union[str, Path]):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        # Sort files for deterministic ordering across systems
        self.fps = tuple(sorted(data_path.glob("*.txt")))
        self.n_paragraphs = {
            fp.name : len(load_paragraphs(fp)) for fp in self.fps
        }

    def _get_fp(self, file_index: int):
        """Get file path by index.

        Args:
            file_index: Index into the list of text files.

        Returns:
            Path object for the requested file.

        Raises:
            ValueError: If file_index is out of bounds.
        """
        if file_index >= len(self.fps):
            raise ValueError(f"Invalid file index; must be less than {len(self.fps)}")
        elif file_index < 0:
            raise ValueError(f"Invalid file index; must be greater than or equal to 0")
        else:
            return self.fps[file_index]

    def get_paragraph_chunk(self, file_index: int, paragraph_range: Optional[Union[slice, int]] = None) -> TextSegment:
        """Get paragraphs from a file by index.

        Args:
            file_index: Index of the file to read.
            paragraph_range: Slice, int, or None for all paragraphs.

        Returns:
            TextSegment with paragraphs and full provenance.
        """
        fp = self._get_fp(file_index)
        paragraphs = load_paragraphs(fp, paragraph_range)

        # Calculate actual indices
        if paragraph_range is None:
            start_idx = 0
            end_idx = len(paragraphs)
        elif isinstance(paragraph_range, int):
            start_idx = paragraph_range
            end_idx = paragraph_range + 1
        elif isinstance(paragraph_range, slice):
            # Handle slice edge cases
            all_paras = load_paragraphs(fp)
            total = len(all_paras)
            start_idx = paragraph_range.start if paragraph_range.start is not None else 0
            end_idx = paragraph_range.stop if paragraph_range.stop is not None else total

            # Handle negative indices
            if start_idx < 0:
                start_idx = max(0, total + start_idx)
            if end_idx < 0:
                end_idx = max(0, total + end_idx)
        else:
            raise ValueError("paragraph_range must be None, int, or slice")

        return TextSegment(
            paragraphs=paragraphs,
            file_index=file_index,
            paragraph_start=start_idx,
            paragraph_end=end_idx,
            file_path=fp
        )

    def iter_paragraph_chunks(self, file_index: int, chunk_size: int, step_size: Optional[int] = None) -> Iterator[TextSegment]:
        """Iterate over chunks of paragraphs from a file.

        Args:
            file_index: Index of the file to read.
            chunk_size: Number of paragraphs per chunk.
            step_size: Number of paragraphs to advance between chunks.
                      Defaults to chunk_size (non-overlapping chunks).

        Yields:
            TextSegment objects, each representing a chunk with provenance.
        """
        if step_size is None:
            step_size = chunk_size

        fp = self._get_fp(file_index)
        paragraphs = load_paragraphs(fp)

        for i in range(0, len(paragraphs), step_size):
            chunk_end = min(i + chunk_size, len(paragraphs))
            chunk = paragraphs[i:chunk_end]

            yield TextSegment(
                paragraphs=chunk,
                file_index=file_index,
                paragraph_start=i,
                paragraph_end=chunk_end,
                file_path=fp
            )

    def sample_segment(self, p_length: int) -> TextSegment:
        """Sample a random segment of consecutive paragraphs.

        Files are weighted by their paragraph count, so longer files
        are more likely to be selected.

        Args:
            p_length: Number of consecutive paragraphs to sample.

        Returns:
            TextSegment with random sample and full provenance.

        Raises:
            ValueError: If p_length exceeds available paragraphs in selected file.
        """
        # Weight by paragraph count
        fp = random.choices(self.fps,
                            weights=[self.n_paragraphs[f.name] for f in self.fps],
                            k=1)[0]

        # Find file_index for this path
        file_index = self.fps.index(fp)

        # Validate p_length
        max_paras = self.n_paragraphs[fp.name]
        if p_length > max_paras:
            raise ValueError(
                f"p_length ({p_length}) exceeds paragraphs in {fp.name} ({max_paras})"
            )

        # Random start position
        p_index = random.randint(0, max_paras - p_length)
        paragraphs = load_paragraphs(fp, slice(p_index, p_index + p_length))

        return TextSegment(
            paragraphs=paragraphs,
            file_index=file_index,
            paragraph_start=p_index,
            paragraph_end=p_index + p_length,
            file_path=fp
        )


if __name__ == "__main__":
    sampler = DataSampler('../data/russell')
    segment = sampler.sample_segment(10)
    print(f"Sampled from file {segment.file_index}: {segment.file_path.name}")
    print(f"Paragraphs {segment.paragraph_start}-{segment.paragraph_end}")
    print(f"\n{segment.text}")
