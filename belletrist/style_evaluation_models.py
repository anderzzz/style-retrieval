"""
Pydantic models for style evaluation experiments.

These models support comparative blind ranking evaluation of text reconstruction
methods. Modified from russell_writes to support configurable method names.
"""
from pydantic import BaseModel, Field
from typing import Literal


class MethodMapping(BaseModel):
    """
    Maps anonymous labels to reconstruction methods for blind evaluation.

    Used to track which text (A, B, C, D) corresponds to which method
    during comparative judging. Supports any method names (not hardcoded).
    """

    text_a: str = Field(..., description="Method assigned to Text A")
    text_b: str = Field(..., description="Method assigned to Text B")
    text_c: str = Field(..., description="Method assigned to Text C")
    text_d: str = Field(..., description="Method assigned to Text D")

    def model_post_init(self, __context):
        """Ensure all 4 methods are unique (no duplicates)."""
        methods = {self.text_a, self.text_b, self.text_c, self.text_d}
        if len(methods) != 4:
            raise ValueError(
                f"Must map 4 unique methods. "
                f"Got {len(methods)} unique values: {methods}"
            )


class StyleJudgmentComparative(BaseModel):
    """
    Structured output for comparative blind ranking of 4 reconstructions.

    Judge ranks 4 reconstructions (labeled A, B, C, D) from 1-4 based on
    stylistic similarity to the original gold standard text.
    """

    ranking_text_a: int = Field(..., ge=1, le=4, description="Rank for Text A (1-4)")
    ranking_text_b: int = Field(..., ge=1, le=4, description="Rank for Text B (1-4)")
    ranking_text_c: int = Field(..., ge=1, le=4, description="Rank for Text C (1-4)")
    ranking_text_d: int = Field(..., ge=1, le=4, description="Rank for Text D (1-4)")
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Judge's confidence level in the rankings"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Chain-of-thought explanation for the ranking decisions"
    )

    def model_post_init(self, __context):
        """Ensure rankings are 1, 2, 3, 4 exactly (no ties, no duplicates)."""
        rankings = {
            self.ranking_text_a,
            self.ranking_text_b,
            self.ranking_text_c,
            self.ranking_text_d
        }
        if rankings != {1, 2, 3, 4}:
            raise ValueError(
                f"Rankings must be exactly [1, 2, 3, 4] with no duplicates. "
                f"Got {sorted([self.ranking_text_a, self.ranking_text_b, self.ranking_text_c, self.ranking_text_d])}"
            )


class MethodMapping5Way(BaseModel):
    """
    Maps anonymous labels to methods for 5-way blind evaluation.

    Used when including the original text as a control alongside 4 reconstruction methods.
    Supports any method names (not hardcoded).
    """

    text_a: str = Field(..., description="Method assigned to Text A")
    text_b: str = Field(..., description="Method assigned to Text B")
    text_c: str = Field(..., description="Method assigned to Text C")
    text_d: str = Field(..., description="Method assigned to Text D")
    text_e: str = Field(..., description="Method assigned to Text E")

    def model_post_init(self, __context):
        """Ensure all 5 labels are unique (no duplicates)."""
        methods = {self.text_a, self.text_b, self.text_c, self.text_d, self.text_e}
        if len(methods) != 5:
            raise ValueError(
                f"Must map 5 unique methods. "
                f"Got {len(methods)} unique values: {methods}"
            )


class StyleJudgmentComparative5Way(BaseModel):
    """
    Structured output for comparative blind ranking of 5 texts.

    Judge ranks 5 texts (labeled A, B, C, D, E) from 1-5 based on
    stylistic similarity to the original gold standard text.

    Use case: Include the original as one of the 5 texts to test judge reliability.
    """

    ranking_text_a: int = Field(..., ge=1, le=5, description="Rank for Text A (1-5)")
    ranking_text_b: int = Field(..., ge=1, le=5, description="Rank for Text B (1-5)")
    ranking_text_c: int = Field(..., ge=1, le=5, description="Rank for Text C (1-5)")
    ranking_text_d: int = Field(..., ge=1, le=5, description="Rank for Text D (1-5)")
    ranking_text_e: int = Field(..., ge=1, le=5, description="Rank for Text E (1-5)")
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Judge's confidence level in the rankings"
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        description="Chain-of-thought explanation for the ranking decisions"
    )

    def model_post_init(self, __context):
        """Ensure rankings are 1, 2, 3, 4, 5 exactly (no ties, no duplicates)."""
        rankings = {
            self.ranking_text_a,
            self.ranking_text_b,
            self.ranking_text_c,
            self.ranking_text_d,
            self.ranking_text_e
        }
        if rankings != {1, 2, 3, 4, 5}:
            raise ValueError(
                f"Rankings must be exactly [1, 2, 3, 4, 5] with no duplicates. "
                f"Got {sorted([self.ranking_text_a, self.ranking_text_b, self.ranking_text_c, self.ranking_text_d, self.ranking_text_e])}"
            )
