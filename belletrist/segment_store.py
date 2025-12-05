"""
Segment storage for curated few-shot examples.

Provides catalog-based retrieval following the Anthropic skills pattern.
Stores exemplary text segments with form/function descriptions and tags.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from belletrist.data_sampler import TextSegment


@dataclass
class SegmentRecord:
    """A segment record from the catalog with all metadata.

    This mirrors the database schema and provides a clean interface
    for working with retrieved segments.
    """
    segment_id: str
    file_index: int
    paragraph_start: int
    paragraph_end: int
    file_name: str
    text: str
    functional_description: str
    formal_description: str
    tags: List[str]
    created_at: str
    source: str

    def to_text_segment(self, data_sampler) -> TextSegment:
        """Recreate TextSegment for use with DataSampler.

        Args:
            data_sampler: DataSampler instance for retrieving from source files

        Returns:
            TextSegment reconstructed from source files using stored provenance
        """
        return data_sampler.get_paragraph_chunk(
            file_index=self.file_index,
            paragraph_range=slice(self.paragraph_start, self.paragraph_end)
        )

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export/display.

        Returns:
            Dictionary with all segment fields
        """
        return {
            'segment_id': self.segment_id,
            'file_index': self.file_index,
            'paragraph_start': self.paragraph_start,
            'paragraph_end': self.paragraph_end,
            'file_name': self.file_name,
            'text': self.text,
            'functional_description': self.functional_description,
            'formal_description': self.formal_description,
            'tags': self.tags,
            'created_at': self.created_at,
            'source': self.source
        }


class SegmentStore:
    """
    SQLite-backed storage for curated text segments.

    Following ResultStore patterns:
    - Row factory for dict-like access
    - Auto-generated sequential IDs (seg_001, seg_002, ...)
    - JSON serialization for tags
    - Provenance stored directly in table

    Design for browsability:
    - Agents can list_all_tags() to see available categories
    - browse_catalog() provides human-readable summaries
    - search_by_tag() enables targeted retrieval

    Example:
        with SegmentStore("segments.db") as store:
            # Save segment
            seg_id = store.save_segment(
                text_segment=segment,
                functional_description="Defines concept clearly",
                formal_description="Topic sentence + examples + conclusion",
                tags=["clear_definition", "pedagogical"]
            )

            # Browse catalog
            catalog = store.browse_catalog(limit=10)

            # Retrieve by ID
            record = store.get_segment(seg_id)
    """

    def __init__(self, db_path: str | Path = "segments.db"):
        """Initialize segment store with database connection.

        Args:
            db_path: Path to SQLite database file. Will be created if doesn't exist.
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Dict-like row access
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        cursor = self.conn.cursor()

        # Create segments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                file_index INTEGER NOT NULL,
                paragraph_start INTEGER NOT NULL,
                paragraph_end INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                text TEXT NOT NULL,
                functional_description TEXT NOT NULL,
                formal_description TEXT NOT NULL,
                tags TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'llm_analysis'
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_file_index ON segments(file_index)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON segments(source)
        """)

        self.conn.commit()

    def _generate_segment_id(self) -> str:
        """Generate next sequential segment ID (seg_001, seg_002, ...).

        Returns:
            Generated segment_id string
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM segments")
        count = cursor.fetchone()[0]
        return f"seg_{count + 1:03d}"

    def save_segment(
        self,
        text_segment: TextSegment,
        functional_description: str,
        formal_description: str,
        tags: List[str],
        source: str = "llm_analysis"
    ) -> str:
        """Save a segment to the catalog.

        Args:
            text_segment: TextSegment with provenance from DataSampler
            functional_description: What the text accomplishes (function)
            formal_description: How the text is structured (form)
            tags: List of descriptive tags (form/function focus)
            source: How this segment was identified (default: "llm_analysis")

        Returns:
            Generated segment_id

        Example:
            segment = sampler.get_paragraph_chunk(0, slice(10, 15))
            seg_id = store.save_segment(
                text_segment=segment,
                functional_description="Introduces counterargument via rhetorical question",
                formal_description="Second-person address, short question + long rebuttal",
                tags=["counterargument", "rhetorical_question", "dialectical"]
            )
        """
        segment_id = self._generate_segment_id()

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO segments (
                segment_id, file_index, paragraph_start, paragraph_end,
                file_name, text, functional_description, formal_description,
                tags, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            segment_id,
            text_segment.file_index,
            text_segment.paragraph_start,
            text_segment.paragraph_end,
            text_segment.file_path.name,
            text_segment.text,
            functional_description,
            formal_description,
            json.dumps(tags),  # Serialize tags as JSON
            source
        ))

        self.conn.commit()
        return segment_id

    def get_segment(self, segment_id: str) -> Optional[SegmentRecord]:
        """Retrieve a segment by ID.

        Args:
            segment_id: Segment identifier (e.g., "seg_001")

        Returns:
            SegmentRecord or None if not found

        Example:
            record = store.get_segment("seg_003")
            if record:
                print(record.functional_description)
                print(record.text)
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM segments WHERE segment_id = ?
        """, (segment_id,))

        row = cursor.fetchone()
        if row is None:
            return None

        return self._row_to_record(row)

    def search_by_tag(
        self,
        tag: str,
        exact_match: bool = True
    ) -> List[SegmentRecord]:
        """Search for segments by tag.

        Args:
            tag: Tag to search for
            exact_match: If True, match exact tag. If False, match substring.

        Returns:
            List of matching SegmentRecords

        Example:
            # Find all segments tagged with "clear_definition"
            segments = store.search_by_tag("clear_definition")

            # Find all segments with "definition" in any tag
            segments = store.search_by_tag("definition", exact_match=False)
        """
        cursor = self.conn.cursor()

        if exact_match:
            # JSON exact match: tag must be in the list
            # SQLite JSON support is limited, so we do post-filtering
            cursor.execute("SELECT * FROM segments")
            rows = cursor.fetchall()

            matches = []
            for row in rows:
                tags = json.loads(row['tags'])
                if tag in tags:
                    matches.append(self._row_to_record(row))
            return matches
        else:
            # Substring match in JSON
            cursor.execute("""
                SELECT * FROM segments WHERE tags LIKE ?
            """, (f'%{tag}%',))
            return [self._row_to_record(row) for row in cursor.fetchall()]

    def list_all_tags(self) -> Dict[str, int]:
        """List all unique tags with counts.

        Returns:
            Dictionary mapping tag -> count, sorted by frequency (descending)

        Example:
            tags = store.list_all_tags()
            # {'clear_definition': 8, 'parallel_structure': 5, 'analogy': 3, ...}

            # Show top 10 most common tags
            for tag, count in list(tags.items())[:10]:
                print(f"{tag}: {count}")
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT tags FROM segments")

        tag_counts = {}
        for row in cursor.fetchall():
            tags = json.loads(row['tags'])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Sort by count descending
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))

    def browse_catalog(
        self,
        limit: Optional[int] = None,
        file_index: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Browse catalog with human-readable summaries.

        Returns minimal info for browsing (not full text).
        Agents can call get_segment() for details.

        Args:
            limit: Maximum number of segments to return
            file_index: Filter by file_index if provided

        Returns:
            List of summary dicts with: segment_id, file_name, functional_description,
            formal_description, tags, paragraph_range

        Example:
            # Browse first 5 segments
            catalog = store.browse_catalog(limit=5)
            for entry in catalog:
                print(f"{entry['segment_id']}: {entry['functional_description']}")
                print(f"  Tags: {', '.join(entry['tags'])}")

            # Browse only segments from file 2
            catalog = store.browse_catalog(file_index=2)
        """
        cursor = self.conn.cursor()

        if file_index is not None:
            query = "SELECT * FROM segments WHERE file_index = ? ORDER BY segment_id"
            cursor.execute(query, (file_index,))
        else:
            query = "SELECT * FROM segments ORDER BY segment_id"
            cursor.execute(query)

        rows = cursor.fetchall()
        if limit:
            rows = rows[:limit]

        summaries = []
        for row in rows:
            summaries.append({
                'segment_id': row['segment_id'],
                'file_name': row['file_name'],
                'functional_description': row['functional_description'],
                'formal_description': row['formal_description'],
                'tags': json.loads(row['tags']),
                'paragraph_range': f"{row['paragraph_start']}-{row['paragraph_end']}"
            })

        return summaries

    def _row_to_record(self, row: sqlite3.Row) -> SegmentRecord:
        """Convert database row to SegmentRecord.

        Args:
            row: SQLite Row object

        Returns:
            SegmentRecord with all fields populated
        """
        return SegmentRecord(
            segment_id=row['segment_id'],
            file_index=row['file_index'],
            paragraph_start=row['paragraph_start'],
            paragraph_end=row['paragraph_end'],
            file_name=row['file_name'],
            text=row['text'],
            functional_description=row['functional_description'],
            formal_description=row['formal_description'],
            tags=json.loads(row['tags']),
            created_at=row['created_at'],
            source=row['source']
        )

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
