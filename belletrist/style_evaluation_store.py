"""
SQLite-backed storage for style evaluation experiments.

This module provides crash-resilient storage for comparative style evaluation
workflows. Modified from russell_writes to support configurable method names.

Workflow: Samples → Flatten → Reconstruct (4 methods × M runs) → Judge comparatively (N times)

Key features:
- Atomic writes: Every LLM call saved immediately (crash resilient)
- Resume support: Check what's done, skip completed work
- Blind evaluation: Stores method mappings for anonymous judging
- Easy export: DataFrame with resolved rankings for analysis
- Judge consistency testing: Multiple judgments per reconstruction set
- Configurable methods: Supports any 4 method names (not hardcoded)

Schema (3 tables):
1. samples: Original texts + flattened content + provenance
2. reconstructions: 4 methods × M reconstruction_runs per sample
3. comparative_judgments: Anonymous rankings + method mappings + confidence
   - Primary key: (sample_id, reconstruction_run, judge_run)
   - Allows multiple judgments of same reconstructions for consistency testing

Usage:
    # With default methods
    store = StyleEvaluationStore(Path("results.db"))

    # With custom methods (e.g., agent_fewshot instead of instructions)
    store = StyleEvaluationStore(
        Path("results.db"),
        methods=['generic', 'fewshot', 'author', 'agent_fewshot']
    )

    # Save sample
    store.save_sample("sample_001", original, flattened, "gpt-4", "File 0, para 50-55")

    # Save reconstructions (reconstruction_run=0)
    store.save_reconstruction("sample_001", run=0, "agent_fewshot", text, "gpt-4")

    # Judge same reconstructions multiple times for consistency testing
    for judge_run in range(3):
        if not store.has_judgment("sample_001", reconstruction_run=0, judge_run=judge_run):
            mapping = store.create_random_mapping(seed=hash(f"sample_001_0_{judge_run}"))
            judgment = ... # Get from LLM
            store.save_judgment("sample_001", reconstruction_run=0, judgment, mapping,
                              "claude-3.5", judge_run=judge_run)

    # Export for analysis
    df = store.to_dataframe()  # Rankings resolved to methods, with judge_run column
"""
from typing import Optional, Literal, List
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import random
import shutil


class StyleEvaluationStore:
    """SQLite-backed crash-resilient storage for style evaluation experiments."""

    def __init__(self, filepath: Path, methods: Optional[List[str]] = None):
        """Initialize store and create schema if needed.

        Args:
            filepath: Path to SQLite database file
            methods: List of 4 method names for this experiment.
                    Default: ['generic', 'fewshot', 'author', 'instructions']
                    Example: ['generic', 'fewshot', 'author', 'agent_fewshot']

        Raises:
            ValueError: If not exactly 4 methods provided, or method names invalid
        """
        self.filepath = filepath

        # Store and validate methods
        self.methods = methods or ['generic', 'fewshot', 'author', 'instructions']

        if len(self.methods) != 4:
            raise ValueError(
                f"Exactly 4 methods required. Got {len(self.methods)}: {self.methods}"
            )

        # Validate method names (must be valid Python identifiers for column names)
        for method in self.methods:
            if not method.isidentifier():
                raise ValueError(
                    f"Invalid method name: '{method}'. "
                    f"Method names must be valid Python identifiers (alphanumeric + underscore, no spaces)."
                )

        # Check for duplicates
        if len(set(self.methods)) != 4:
            raise ValueError(
                f"Method names must be unique. Got duplicates in: {self.methods}"
            )

        self.conn = sqlite3.connect(filepath)
        self.conn.row_factory = sqlite3.Row  # Dict-like row access

        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")

        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist (with dynamic CHECK constraints)."""
        # Table 1: Test samples and flattened content
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                original_text TEXT NOT NULL,
                flattened_content TEXT NOT NULL,
                flattening_model TEXT NOT NULL,
                source_info TEXT
            )
        """)

        # Table 2: Reconstructions (4 per sample per run)
        # Build CHECK constraint dynamically based on configured methods
        methods_constraint = ', '.join(f"'{m}'" for m in self.methods)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS reconstructions (
                sample_id TEXT NOT NULL,
                run INTEGER NOT NULL,
                method TEXT NOT NULL CHECK(method IN ({methods_constraint})),
                reconstructed_text TEXT NOT NULL,
                reconstruction_model TEXT NOT NULL,
                PRIMARY KEY (sample_id, run, method),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)

        # Table 3: Comparative judgments (multiple judgments per sample per reconstruction_run)
        # Build CHECK constraints dynamically for method mappings
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS comparative_judgments (
                sample_id TEXT NOT NULL,
                reconstruction_run INTEGER NOT NULL,
                judge_run INTEGER NOT NULL DEFAULT 0,

                -- Anonymous rankings (as judge returned them)
                ranking_text_a INTEGER NOT NULL CHECK(ranking_text_a BETWEEN 1 AND 4),
                ranking_text_b INTEGER NOT NULL CHECK(ranking_text_b BETWEEN 1 AND 4),
                ranking_text_c INTEGER NOT NULL CHECK(ranking_text_c BETWEEN 1 AND 4),
                ranking_text_d INTEGER NOT NULL CHECK(ranking_text_d BETWEEN 1 AND 4),

                -- Method mapping (which label = which method)
                method_text_a TEXT NOT NULL CHECK(method_text_a IN ({methods_constraint})),
                method_text_b TEXT NOT NULL CHECK(method_text_b IN ({methods_constraint})),
                method_text_c TEXT NOT NULL CHECK(method_text_c IN ({methods_constraint})),
                method_text_d TEXT NOT NULL CHECK(method_text_d IN ({methods_constraint})),

                -- Judgment metadata
                confidence TEXT NOT NULL CHECK(confidence IN ('high', 'medium', 'low')),
                reasoning TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                PRIMARY KEY (sample_id, reconstruction_run, judge_run),
                FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
            )
        """)

        self.conn.commit()

    # ==========================================================================
    # Sample Management
    # ==========================================================================

    def save_sample(
        self,
        sample_id: str,
        original_text: str,
        flattened_content: str,
        flattening_model: str,
        source_info: Optional[str] = None
    ):
        """Store a text sample with its flattened content.

        Args:
            sample_id: Unique identifier for this sample
            original_text: The original gold standard text
            flattened_content: Content-only summary (output of style flattening)
            flattening_model: Model used for flattening (e.g., 'anthropic/claude-haiku')
            source_info: Optional provenance (e.g., "File 0, para 50-55")
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO samples
            VALUES (?, ?, ?, ?, ?)
        """, (sample_id, original_text, flattened_content, flattening_model, source_info))
        self.conn.commit()

    def get_sample(self, sample_id: str) -> Optional[dict]:
        """Retrieve a sample by ID.

        Args:
            sample_id: Sample identifier

        Returns:
            Dictionary with keys: sample_id, original_text, flattened_content,
            flattening_model, source_info. Returns None if not found
        """
        row = self.conn.execute(
            "SELECT * FROM samples WHERE sample_id=?",
            (sample_id,)
        ).fetchone()

        if not row:
            return None

        return {
            'sample_id': row['sample_id'],
            'original_text': row['original_text'],
            'flattened_content': row['flattened_content'],
            'flattening_model': row['flattening_model'],
            'source_info': row['source_info']
        }

    def list_samples(self) -> list[str]:
        """Get all sample IDs in insertion order.

        Returns:
            List of sample IDs
        """
        rows = self.conn.execute(
            "SELECT sample_id FROM samples ORDER BY rowid"
        ).fetchall()
        return [row['sample_id'] for row in rows]

    # ==========================================================================
    # Reconstruction Management
    # ==========================================================================

    def save_reconstruction(
        self,
        sample_id: str,
        run: int,
        method: str,
        reconstructed_text: str,
        model: str
    ):
        """Save a reconstruction result.

        Args:
            sample_id: ID of the sample being reconstructed
            run: Run number (0 to M-1 for M stochastic runs)
            method: Reconstruction method (must be one of the configured methods)
            reconstructed_text: The reconstructed text
            model: Model used for reconstruction (e.g., 'mistral-large-2411')

        Raises:
            ValueError: If sample_id doesn't exist or method not in configured methods
        """
        # Verify sample exists
        if not self.get_sample(sample_id):
            raise ValueError(
                f"Sample '{sample_id}' not found. Save sample first with save_sample()."
            )

        # Verify method is valid
        if method not in self.methods:
            raise ValueError(
                f"Method '{method}' not in configured methods: {self.methods}"
            )

        self.conn.execute("""
            INSERT OR REPLACE INTO reconstructions
            VALUES (?, ?, ?, ?, ?)
        """, (sample_id, run, method, reconstructed_text, model))
        self.conn.commit()

    def has_reconstruction(self, sample_id: str, run: int, method: str) -> bool:
        """Check if a reconstruction exists.

        Args:
            sample_id: Sample identifier
            run: Run number
            method: Reconstruction method

        Returns:
            True if reconstruction exists, False otherwise
        """
        row = self.conn.execute("""
            SELECT 1 FROM reconstructions
            WHERE sample_id=? AND run=? AND method=?
        """, (sample_id, run, method)).fetchone()
        return row is not None

    def get_reconstructions(self, sample_id: str, run: int) -> dict[str, str]:
        """Get all 4 reconstructions for a sample/run.

        Args:
            sample_id: Sample identifier
            run: Run number

        Returns:
            Dictionary mapping method to reconstructed text
            Example: {'generic': '...', 'fewshot': '...', 'author': '...', 'agent_fewshot': '...'}
        """
        rows = self.conn.execute("""
            SELECT method, reconstructed_text
            FROM reconstructions
            WHERE sample_id=? AND run=?
        """, (sample_id, run)).fetchall()

        return {row['method']: row['reconstructed_text'] for row in rows}

    # ==========================================================================
    # Judgment Management
    # ==========================================================================

    def save_judgment(
        self,
        sample_id: str,
        reconstruction_run: int,
        judgment: 'StyleJudgmentComparative',
        mapping: 'MethodMapping',
        judge_model: str,
        judge_run: int = 0
    ):
        """Save a comparative judgment with method mapping.

        Args:
            sample_id: Sample identifier
            reconstruction_run: Reconstruction run number
            judgment: StyleJudgmentComparative instance from LLM
            mapping: MethodMapping showing which label corresponds to which method
            judge_model: Model used for judging (e.g., 'claude-sonnet-4-5')
            judge_run: Judge run number (default 0). Use > 0 for consistency testing
                      (multiple judgments of same reconstructions)

        Raises:
            ValueError: If sample doesn't exist or not all 4 reconstructions exist
        """
        # Verify sample exists
        if not self.get_sample(sample_id):
            raise ValueError(
                f"Sample '{sample_id}' not found. Save sample first."
            )

        # Verify all 4 reconstructions exist
        reconstructions = self.get_reconstructions(sample_id, reconstruction_run)
        if len(reconstructions) != 4:
            missing = set(self.methods) - set(reconstructions.keys())
            raise ValueError(
                f"Missing reconstructions for sample '{sample_id}', reconstruction_run {reconstruction_run}: {missing}"
            )

        # Save judgment
        timestamp = datetime.now().isoformat()
        self.conn.execute("""
            INSERT OR REPLACE INTO comparative_judgments
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id,
            reconstruction_run,
            judge_run,
            judgment.ranking_text_a,
            judgment.ranking_text_b,
            judgment.ranking_text_c,
            judgment.ranking_text_d,
            mapping.text_a,
            mapping.text_b,
            mapping.text_c,
            mapping.text_d,
            judgment.confidence,
            judgment.reasoning,
            judge_model,
            timestamp
        ))
        self.conn.commit()

    def has_judgment(self, sample_id: str, reconstruction_run: int, judge_run: int = 0) -> bool:
        """Check if a judgment exists for a sample/reconstruction_run/judge_run.

        Args:
            sample_id: Sample identifier
            reconstruction_run: Reconstruction run number
            judge_run: Judge run number (default 0)

        Returns:
            True if judgment exists, False otherwise
        """
        row = self.conn.execute("""
            SELECT 1 FROM comparative_judgments
            WHERE sample_id=? AND reconstruction_run=? AND judge_run=?
        """, (sample_id, reconstruction_run, judge_run)).fetchone()
        return row is not None

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def create_random_mapping(self, seed: Optional[int] = None) -> 'MethodMapping':
        """Generate random label-to-method mapping for blind evaluation.

        Randomly assigns the 4 configured methods to labels A, B, C, D. Useful for
        eliminating position bias by varying order across samples/runs.

        Args:
            seed: Optional random seed for deterministic mapping

        Returns:
            MethodMapping instance with randomized assignments

        Example:
            >>> store = StyleEvaluationStore(Path("test.db"), methods=['generic', 'fewshot', 'author', 'agent_fewshot'])
            >>> mapping = store.create_random_mapping(seed=42)
            >>> mapping.text_a  # Might be 'agent_fewshot'
            >>> mapping.text_b  # Might be 'generic'
        """
        from belletrist.style_evaluation_models import MethodMapping

        methods = self.methods.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(methods)

        return MethodMapping(
            text_a=methods[0],
            text_b=methods[1],
            text_c=methods[2],
            text_d=methods[3]
        )

    def get_incomplete_work(self, n_runs: int) -> list[tuple[str, int]]:
        """Find (sample_id, run) pairs that need judgments.

        Useful for resume support: identify which work is incomplete.

        Args:
            n_runs: Expected number of runs per sample

        Returns:
            List of (sample_id, run) tuples that don't have judgments yet
        """
        incomplete = []
        for sample_id in self.list_samples():
            for run in range(n_runs):
                if not self.has_judgment(sample_id, run):
                    incomplete.append((sample_id, run))
        return incomplete

    # ==========================================================================
    # Export Methods
    # ==========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export all judgments to DataFrame with method-resolved rankings.

        Resolves anonymous rankings (text_a, text_b, etc.) to actual method
        rankings (ranking_<method>, e.g. ranking_agent_fewshot) for analysis.

        Returns:
            DataFrame with columns:
                - sample_id: Sample identifier
                - reconstruction_run: Reconstruction run number
                - judge_run: Judge run number
                - ranking_<method1>: Rank of method1 (1-4) - one column per configured method
                - ranking_<method2>: Rank of method2 (1-4)
                - ranking_<method3>: Rank of method3 (1-4)
                - ranking_<method4>: Rank of method4 (1-4)
                - confidence: Judge confidence (high/medium/low)
                - reasoning: Judge's explanation
                - judge_model: Model used for judging
                - timestamp: When judgment was made
        """
        rows = self.conn.execute("""
            SELECT * FROM comparative_judgments ORDER BY sample_id, reconstruction_run, judge_run
        """).fetchall()

        records = []
        for row in rows:
            # Build mapping: method -> ranking
            method_rankings = {}
            for label in ['a', 'b', 'c', 'd']:
                method = row[f'method_text_{label}']
                ranking = row[f'ranking_text_{label}']
                method_rankings[method] = ranking

            # Build record with dynamic method columns
            record = {
                'sample_id': row['sample_id'],
                'reconstruction_run': row['reconstruction_run'],
                'judge_run': row['judge_run'],
            }

            # Add ranking columns for each configured method
            for method in self.methods:
                record[f'ranking_{method}'] = method_rankings[method]

            # Add metadata columns
            record.update({
                'confidence': row['confidence'],
                'reasoning': row['reasoning'],
                'judge_model': row['judge_model'],
                'timestamp': row['timestamp']
            })

            records.append(record)

        return pd.DataFrame(records)

    def to_csv(self, output_path: Path):
        """Export judgments to CSV file.

        Args:
            output_path: Path to write CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)

    # ==========================================================================
    # Stats Methods
    # ==========================================================================

    def get_stats(self) -> dict:
        """Get statistics about the evaluation progress.

        Returns:
            Dictionary with:
                - n_samples: Total samples
                - n_reconstructions: Total reconstructions saved
                - n_judgments: Total judgments saved
                - configured_methods: List of method names for this experiment
        """
        n_samples = len(self.list_samples())

        n_reconstructions = self.conn.execute(
            "SELECT COUNT(*) as count FROM reconstructions"
        ).fetchone()['count']

        n_judgments = self.conn.execute(
            "SELECT COUNT(*) as count FROM comparative_judgments"
        ).fetchone()['count']

        return {
            'n_samples': n_samples,
            'n_reconstructions': n_reconstructions,
            'n_judgments': n_judgments,
            'configured_methods': self.methods
        }

    # ==========================================================================
    # Reset Methods
    # ==========================================================================

    def reset(self, scope: Literal['all', 'reconstructions_and_judgments', 'judgments_only'] = 'all'):
        """Clear data from the store with hierarchical scope control.

        The data has a clear dependency hierarchy:
            samples (base)
              ↓
            reconstructions (depends on samples via FK)
              ↓
            comparative_judgments (depends on samples via FK)

        Valid reset operations must respect this hierarchy - you cannot delete
        upstream data (e.g., samples) while preserving downstream data (e.g., judgments)
        without violating foreign key constraints.

        Args:
            scope: Reset scope controlling which tables to clear:
                - 'all': Delete everything (samples, reconstructions, judgments)
                - 'reconstructions_and_judgments': Keep samples, delete reconstructions + judgments
                - 'judgments_only': Keep samples + reconstructions, delete only judgments

        Warning:
            This operation is irreversible. All data in the selected scope
            will be permanently deleted.

        Design Notes:
            - Deletion order matters: must delete child tables before parents
            - Cannot delete samples while keeping reconstructions (would violate FKs)
            - Cannot delete reconstructions while keeping judgments (would violate FKs)

        Example:
            >>> store.reset('judgments_only')  # Re-run judging, keep reconstructions
            >>> store.reset('reconstructions_and_judgments')  # Re-run reconstruction + judging
            >>> store.reset('all')  # Fresh start
        """
        if scope == 'all':
            # Delete everything in reverse dependency order
            self.conn.execute("DELETE FROM comparative_judgments")
            self.conn.execute("DELETE FROM reconstructions")
            self.conn.execute("DELETE FROM samples")
        elif scope == 'reconstructions_and_judgments':
            # Keep samples, delete everything downstream
            self.conn.execute("DELETE FROM comparative_judgments")
            self.conn.execute("DELETE FROM reconstructions")
        elif scope == 'judgments_only':
            # Keep samples and reconstructions, delete only judgments
            self.conn.execute("DELETE FROM comparative_judgments")
        else:
            raise ValueError(
                f"Invalid scope: {scope}. "
                f"Must be 'all', 'reconstructions_and_judgments', or 'judgments_only'"
            )

        self.conn.commit()

    def copy_to(self, destination: Path) -> 'StyleEvaluationStore':
        """Create a copy of this database at a new location.

        Useful for running multiple experiments with the same flattened samples
        but different reconstruction/judging models.

        Args:
            destination: Path for the new database file

        Returns:
            New StyleEvaluationStore instance pointing to the copied database

        Raises:
            FileExistsError: If destination already exists

        Example:
            >>> store1 = StyleEvaluationStore(Path("exp1.db"))
            >>> # ... run flattening, reconstruction, judging
            >>> store2 = store1.copy_to(Path("exp2.db"))
            >>> store2.reset('reconstructions_and_judgments')
            >>> # ... run new reconstruction/judging with different models
        """
        destination = Path(destination)

        if destination.exists():
            raise FileExistsError(
                f"Destination {destination} already exists. "
                f"Delete it first or choose a different path."
            )

        # Close current connection to ensure all writes are flushed
        self.conn.close()

        # Copy the database file
        shutil.copy2(self.filepath, destination)

        # Reopen this connection
        self.conn = sqlite3.connect(self.filepath)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        # Return new instance pointing to the copy (with same methods config)
        return StyleEvaluationStore(destination, methods=self.methods)

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
