"""
Cross-model/cross-method comparison utilities.

This module enables comparing reconstructions across different model+method
combinations by pulling from multiple StyleEvaluationStore databases.

Workflow:
1. Define combinations: [(db_path, method, label), ...]
2. Pull reconstructions from each database
3. Run random k-way comparisons (sample 4 combinations, judge ranks them)
4. Store results for Bradley-Terry analysis

Example:
    combinations = [
        (Path("eval_mistral.db"), "fewshot", "mistral_fewshot"),
        (Path("eval_mistral.db"), "agent_holistic", "mistral_agent"),
        (Path("eval_kimi.db"), "fewshot", "kimi_fewshot"),
        (Path("eval_kimi.db"), "agent_holistic", "kimi_agent"),
    ]

    comparer = CrossModelComparisonStore(
        output_db=Path("cross_comparison.db"),
        combinations=combinations
    )

    # Pull all reconstructions
    comparer.load_reconstructions(sample_ids=['sample_000', 'sample_001'])

    # Run 8 random 4-way comparisons per sample
    comparer.run_comparisons(
        judge_llm=judge_llm,
        prompt_maker=prompt_maker,
        n_comparisons_per_sample=8
    )

    # Export for Bradley-Terry analysis
    df = comparer.to_dataframe()
"""
from typing import List, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import sqlite3
import random
import pandas as pd
from datetime import datetime

from belletrist import StyleEvaluationStore


@dataclass
class Combination:
    """A (database, method) combination to compare.

    Attributes:
        db_path: Path to StyleEvaluationStore database
        method: Reconstruction method name (e.g., 'fewshot')
        label: Unique label for this combination (e.g., 'mistral_fewshot')
        reconstruction_run: Which reconstruction run to use (default 0)
    """
    db_path: Path
    method: str
    label: str
    reconstruction_run: int = 0

    def __post_init__(self):
        """Validate label is a valid identifier."""
        if not self.label.replace('_', '').isalnum():
            raise ValueError(
                f"Label must be alphanumeric+underscore: {self.label}"
            )


class CrossModelComparisonStore:
    """Storage and orchestration for cross-model comparisons.

    Pulls reconstructions from multiple source databases and runs
    random k-way comparative judgments.
    """

    def __init__(self, output_db: Path, combinations: List[Combination]):
        """Initialize comparison store.

        Args:
            output_db: Path to output SQLite database
            combinations: List of Combination objects to compare

        Raises:
            ValueError: If duplicate labels or less than 4 combinations
        """
        self.output_db = output_db
        self.combinations = combinations

        # Validate
        if len(combinations) < 4:
            raise ValueError(
                f"Need at least 4 combinations for 4-way comparison. Got {len(combinations)}"
            )

        labels = [c.label for c in combinations]
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate labels found: {labels}")

        # Initialize database
        self.conn = sqlite3.connect(output_db)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        """Create tables for cross-model comparison."""

        # Table 1: Track which combinations are being compared
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS combinations (
                label TEXT PRIMARY KEY,
                db_path TEXT NOT NULL,
                method TEXT NOT NULL,
                reconstruction_run INTEGER NOT NULL
            )
        """)

        # Store combination metadata
        for combo in self.combinations:
            self.conn.execute("""
                INSERT OR REPLACE INTO combinations VALUES (?, ?, ?, ?)
            """, (combo.label, str(combo.db_path), combo.method, combo.reconstruction_run))

        # Table 2: Reconstructions pulled from source databases
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reconstructions (
                sample_id TEXT NOT NULL,
                label TEXT NOT NULL,
                original_text TEXT NOT NULL,
                reconstructed_text TEXT NOT NULL,
                source_db TEXT NOT NULL,
                method TEXT NOT NULL,
                reconstruction_model TEXT NOT NULL,
                PRIMARY KEY (sample_id, label),
                FOREIGN KEY (label) REFERENCES combinations(label)
            )
        """)

        # Table 3: Comparative judgments (4-way rankings)
        # Similar to StyleEvaluationStore but with dynamic labels
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS comparative_judgments (
                judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                comparison_run INTEGER NOT NULL,

                -- Anonymous rankings (as judge returned them)
                ranking_text_a INTEGER NOT NULL CHECK(ranking_text_a BETWEEN 1 AND 4),
                ranking_text_b INTEGER NOT NULL CHECK(ranking_text_b BETWEEN 1 AND 4),
                ranking_text_c INTEGER NOT NULL CHECK(ranking_text_c BETWEEN 1 AND 4),
                ranking_text_d INTEGER NOT NULL CHECK(ranking_text_d BETWEEN 1 AND 4),

                -- Label mapping (which letter = which combination label)
                label_text_a TEXT NOT NULL,
                label_text_b TEXT NOT NULL,
                label_text_c TEXT NOT NULL,
                label_text_d TEXT NOT NULL,

                -- Judgment metadata
                confidence TEXT NOT NULL CHECK(confidence IN ('high', 'medium', 'low')),
                reasoning TEXT NOT NULL,
                judge_model TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                FOREIGN KEY (label_text_a) REFERENCES combinations(label),
                FOREIGN KEY (label_text_b) REFERENCES combinations(label),
                FOREIGN KEY (label_text_c) REFERENCES combinations(label),
                FOREIGN KEY (label_text_d) REFERENCES combinations(label)
            )
        """)

        self.conn.commit()

    # ==========================================================================
    # Loading Reconstructions
    # ==========================================================================

    def load_reconstructions(
        self,
        sample_ids: Optional[List[str]] = None
    ):
        """Load reconstructions from source databases.

        Args:
            sample_ids: List of sample IDs to load. If None, loads all samples
                       found in first source database.

        Raises:
            ValueError: If sample not found in source database or original texts mismatch
        """
        # Group combinations by database for efficient loading
        db_combinations: Dict[Path, List[Combination]] = {}
        for combo in self.combinations:
            if combo.db_path not in db_combinations:
                db_combinations[combo.db_path] = []
            db_combinations[combo.db_path].append(combo)

        # If no sample_ids specified, get from first database
        if sample_ids is None:
            first_db = list(db_combinations.keys())[0]
            with StyleEvaluationStore(first_db) as store:
                sample_ids = store.list_samples()
            print(f"Loading all {len(sample_ids)} samples from source databases")

        # Track original texts to verify consistency
        original_texts: Dict[str, str] = {}

        # Load from each database
        for db_path, combos in db_combinations.items():
            print(f"\nLoading from {db_path.name}:")

            with StyleEvaluationStore(db_path) as store:
                for sample_id in sample_ids:
                    # Get sample
                    sample = store.get_sample(sample_id)
                    if not sample:
                        raise ValueError(
                            f"Sample '{sample_id}' not found in {db_path}"
                        )

                    # Verify original text consistency across databases
                    if sample_id in original_texts:
                        if original_texts[sample_id] != sample['original_text']:
                            raise ValueError(
                                f"Original text mismatch for '{sample_id}' "
                                f"between databases"
                            )
                    else:
                        original_texts[sample_id] = sample['original_text']

                    # Load reconstructions for each combination
                    for combo in combos:
                        if not store.has_reconstruction(
                            sample_id, combo.reconstruction_run, combo.method
                        ):
                            raise ValueError(
                                f"Reconstruction not found: {sample_id}, "
                                f"run={combo.reconstruction_run}, method={combo.method} "
                                f"in {db_path}"
                            )

                        reconstructions = store.get_reconstructions(
                            sample_id, combo.reconstruction_run
                        )
                        reconstructed_text = reconstructions[combo.method]

                        # Get model name
                        row = store.conn.execute("""
                            SELECT reconstruction_model FROM reconstructions
                            WHERE sample_id=? AND run=? AND method=?
                        """, (sample_id, combo.reconstruction_run, combo.method)).fetchone()

                        reconstruction_model = row['reconstruction_model']

                        # Save to comparison database
                        self.conn.execute("""
                            INSERT OR REPLACE INTO reconstructions VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            sample_id,
                            combo.label,
                            sample['original_text'],
                            reconstructed_text,
                            str(db_path),
                            combo.method,
                            reconstruction_model
                        ))

                        print(f"  ✓ {combo.label:30s} - {sample_id}")

        self.conn.commit()
        print(f"\n✓ Loaded {len(sample_ids)} samples × {len(self.combinations)} combinations")

    def get_loaded_samples(self) -> List[str]:
        """Get list of sample IDs that have been loaded.

        Returns:
            List of sample IDs with reconstructions loaded
        """
        rows = self.conn.execute("""
            SELECT DISTINCT sample_id FROM reconstructions ORDER BY sample_id
        """).fetchall()
        return [row['sample_id'] for row in rows]

    def get_reconstruction(self, sample_id: str, label: str) -> Optional[dict]:
        """Get a specific reconstruction.

        Args:
            sample_id: Sample identifier
            label: Combination label

        Returns:
            Dict with keys: original_text, reconstructed_text, method, reconstruction_model
            Returns None if not found
        """
        row = self.conn.execute("""
            SELECT * FROM reconstructions WHERE sample_id=? AND label=?
        """, (sample_id, label)).fetchone()

        if not row:
            return None

        return {
            'original_text': row['original_text'],
            'reconstructed_text': row['reconstructed_text'],
            'method': row['method'],
            'reconstruction_model': row['reconstruction_model']
        }

    # ==========================================================================
    # Running Comparisons
    # ==========================================================================

    def run_comparisons(
        self,
        judge_llm,
        prompt_maker,
        n_comparisons_per_sample: int = 8,
        seed: Optional[int] = None
    ):
        """Run random 4-way comparisons.

        For each sample, randomly sample k sets of 4 combinations,
        present them to the judge in random order, and store rankings.

        Args:
            judge_llm: LLM instance for judging (should support complete_with_schema)
            prompt_maker: PromptMaker instance
            n_comparisons_per_sample: Number of random 4-way comparisons per sample
            seed: Random seed for reproducibility

        Raises:
            ValueError: If fewer than n_comparisons_per_sample * 4 unique combinations
        """
        from belletrist.prompts import StyleJudgeComparativeConfig
        from belletrist.style_evaluation_models import StyleJudgmentComparative

        if seed is not None:
            random.seed(seed)

        # Validate we have enough combinations
        n_combos = len(self.combinations)
        if n_combos < 4:
            raise ValueError(f"Need at least 4 combinations. Got {n_combos}")

        samples = self.get_loaded_samples()
        total_comparisons = len(samples) * n_comparisons_per_sample

        print(f"=== Running {n_comparisons_per_sample} comparisons × {len(samples)} samples = {total_comparisons} total ===\n")

        for sample_id in samples:
            print(f"\n{sample_id}:")

            # Get original text (same for all combinations)
            first_label = self.combinations[0].label
            original_text = self.get_reconstruction(sample_id, first_label)['original_text']

            for comparison_run in range(n_comparisons_per_sample):
                # Check if already done
                if self._has_judgment(sample_id, comparison_run):
                    print(f"  Comparison {comparison_run}: ✓ Already done (skipping)")
                    continue

                print(f"  Comparison {comparison_run}: Sampling 4 combinations...", end=" ")

                # Randomly sample 4 combinations
                sampled_combos = random.sample(self.combinations, k=4)
                sampled_labels = [c.label for c in sampled_combos]

                # Randomly assign to A/B/C/D
                random.shuffle(sampled_labels)
                label_mapping = {
                    'text_a': sampled_labels[0],
                    'text_b': sampled_labels[1],
                    'text_c': sampled_labels[2],
                    'text_d': sampled_labels[3]
                }

                # Get reconstructed texts
                texts = {
                    letter: self.get_reconstruction(sample_id, label)['reconstructed_text']
                    for letter, label in label_mapping.items()
                }

                print(f"[{', '.join(sampled_labels)}]")
                print(f"    Judging...", end=" ")

                # Build judge prompt
                judge_config = StyleJudgeComparativeConfig(
                    original_text=original_text,
                    reconstruction_text_a=texts['text_a'],
                    reconstruction_text_b=texts['text_b'],
                    reconstruction_text_c=texts['text_c'],
                    reconstruction_text_d=texts['text_d']
                )
                judge_prompt = prompt_maker.render(judge_config)

                # Get judgment
                try:
                    response = judge_llm.complete_with_schema(
                        judge_prompt,
                        StyleJudgmentComparative
                    )
                    judgment = response.content

                    # Save judgment
                    self._save_judgment(
                        sample_id=sample_id,
                        comparison_run=comparison_run,
                        judgment=judgment,
                        label_mapping=label_mapping,
                        judge_model=response.model
                    )

                    print(f"✓ (confidence: {judgment.confidence})")

                except Exception as e:
                    print(f"✗ Error: {e}")

        n_judgments = self.conn.execute(
            "SELECT COUNT(*) as count FROM comparative_judgments"
        ).fetchone()['count']

        print(f"\n✓ Completed {n_judgments} total judgments")

    def _has_judgment(self, sample_id: str, comparison_run: int) -> bool:
        """Check if judgment exists."""
        row = self.conn.execute("""
            SELECT 1 FROM comparative_judgments
            WHERE sample_id=? AND comparison_run=?
        """, (sample_id, comparison_run)).fetchone()
        return row is not None

    def _save_judgment(
        self,
        sample_id: str,
        comparison_run: int,
        judgment: 'StyleJudgmentComparative',
        label_mapping: dict,
        judge_model: str
    ):
        """Save a comparative judgment."""
        timestamp = datetime.now().isoformat()

        self.conn.execute("""
            INSERT INTO comparative_judgments (
                sample_id, comparison_run,
                ranking_text_a, ranking_text_b, ranking_text_c, ranking_text_d,
                label_text_a, label_text_b, label_text_c, label_text_d,
                confidence, reasoning, judge_model, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            sample_id,
            comparison_run,
            judgment.ranking_text_a,
            judgment.ranking_text_b,
            judgment.ranking_text_c,
            judgment.ranking_text_d,
            label_mapping['text_a'],
            label_mapping['text_b'],
            label_mapping['text_c'],
            label_mapping['text_d'],
            judgment.confidence,
            judgment.reasoning,
            judge_model,
            timestamp
        ))
        self.conn.commit()

    # ==========================================================================
    # Export and Analysis
    # ==========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Export judgments to DataFrame for Bradley-Terry analysis.

        Returns:
            DataFrame with one row per judgment containing:
                - sample_id: Sample identifier
                - comparison_run: Which random 4-way comparison
                - label_text_{a,b,c,d}: Which combination was shown as each text
                - ranking_text_{a,b,c,d}: Rank assigned to each text (1-4)
                - confidence: Judge confidence
                - reasoning: Judge explanation
                - judge_model: Model used
                - timestamp: When judged
        """
        rows = self.conn.execute("""
            SELECT * FROM comparative_judgments
            ORDER BY sample_id, comparison_run
        """).fetchall()

        records = []
        for row in rows:
            record = {
                'sample_id': row['sample_id'],
                'comparison_run': row['comparison_run'],
                'label_text_a': row['label_text_a'],
                'label_text_b': row['label_text_b'],
                'label_text_c': row['label_text_c'],
                'label_text_d': row['label_text_d'],
                'ranking_text_a': row['ranking_text_a'],
                'ranking_text_b': row['ranking_text_b'],
                'ranking_text_c': row['ranking_text_c'],
                'ranking_text_d': row['ranking_text_d'],
                'confidence': row['confidence'],
                'reasoning': row['reasoning'],
                'judge_model': row['judge_model'],
                'timestamp': row['timestamp']
            }
            records.append(record)

        return pd.DataFrame(records)

    def to_bradley_terry_format(self) -> pd.DataFrame:
        """Convert judgments to Bradley-Terry input format.

        Converts 4-way rankings into pairwise preference data suitable
        for Bradley-Terry model fitting.

        Returns:
            DataFrame with columns:
                - winner: Label of preferred combination
                - loser: Label of less-preferred combination
                - sample_id: Source sample
                - comparison_run: Which comparison

        Example:
            If judgment ranked: A=1, B=3, C=2, D=4
            Generates pairs: (A>B), (A>C), (A>D), (C>B), (C>D), (B>D)
        """
        df = self.to_dataframe()

        pairs = []
        for _, row in df.iterrows():
            # Build label->rank mapping
            rankings = {
                row['label_text_a']: row['ranking_text_a'],
                row['label_text_b']: row['ranking_text_b'],
                row['label_text_c']: row['ranking_text_c'],
                row['label_text_d']: row['ranking_text_d']
            }

            # Generate all pairwise preferences
            labels = list(rankings.keys())
            for i, label1 in enumerate(labels):
                for label2 in labels[i+1:]:
                    rank1 = rankings[label1]
                    rank2 = rankings[label2]

                    if rank1 < rank2:  # Lower rank = better
                        winner, loser = label1, label2
                    else:
                        winner, loser = label2, label1

                    pairs.append({
                        'winner': winner,
                        'loser': loser,
                        'sample_id': row['sample_id'],
                        'comparison_run': row['comparison_run']
                    })

        return pd.DataFrame(pairs)

    def get_stats(self) -> dict:
        """Get comparison statistics.

        Returns:
            Dictionary with:
                - n_combinations: Number of combinations being compared
                - n_samples: Number of samples loaded
                - n_judgments: Number of comparative judgments completed
                - combinations: List of combination labels
        """
        n_judgments = self.conn.execute(
            "SELECT COUNT(*) as count FROM comparative_judgments"
        ).fetchone()['count']

        return {
            'n_combinations': len(self.combinations),
            'n_samples': len(self.get_loaded_samples()),
            'n_judgments': n_judgments,
            'combinations': [c.label for c in self.combinations]
        }

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
