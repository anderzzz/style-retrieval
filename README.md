# Style Retrieval and Evaluation Framework

A systematic framework for analyzing, retrieving, and reconstructing literary style using Large Language Models (LLMs). This project provides tools for extracting exemplary passages from texts, building searchable catalogs of stylistic techniques, and evaluating how well different prompting methods can reconstruct an author's distinctive voice.

**📖 Blog Post**: [Read about the experimental design and findings](#) *(Coming soon)*

## Quick Start for New Users

**First time here?** Start with these three steps:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Set API keys**: Export your OpenAI/Anthropic/Mistral keys
3. **Run the main evaluation**: Open `style_evaluation_statistical.ipynb` and run all cells

The notebook will guide you through neutralizing text samples, reconstructing them with different methods, and comparing the results. See [Installation](#installation) and [Notebooks](#notebooks) for details.

## Key Results

Our experiments with Bertrand Russell's prose (10 samples, 2 runs each, 4 LLMs) show:

| Method | Mean Rank | % Win (1st place) | % Top-2 |
|--------|-----------|-------------------|---------|
| **Agent Statistical** | **1.74** | 48% | 85% |
| **Fewshot** | **1.94** | 33% | 78% |
| Author | 2.65 | 18% | 31% |
| Generic | 3.67 | 1% | 6% |

**Key findings:**
- **Agent Statistical** (retrieval from curated catalog) performs best overall
- **Fewshot** is extremely competitive (only 0.20 behind) and wins on Mistral
- **Author name prompting** alone is surprisingly weak
- **Generic baseline** consistently fails (never wins, rarely ranks in top 2)

📊 See full analysis in [`method_performance_summary.md`](method_performance_summary.md)

## Overview

The framework addresses a core challenge in computational stylistics: **Can LLMs capture and reproduce an author's distinctive style?** To answer this, we:

1. **Extract exemplary passages** from source texts, annotated with craft moves and teaching notes
2. **Build searchable catalogs** of these passages, tagged for retrieval
3. **Evaluate reconstruction methods** by comparing how well different prompting strategies can recreate the original style from neutralized content
4. **Compare LLM performance** across different models and prompting approaches

The included experiments use prose by Bertrand Russell as test data, but the framework is designed to work with any author's writing.

## Key Features

### Experimental Features
- **Segment Analysis**: LLM-powered identification of exemplary passages demonstrating teachable craft moves
- **Style Neutralization**: Rewrites texts in bland journalistic prose while preserving argumentative structure
- **Multiple Reconstruction Methods**: Generic baseline, few-shot learning, author name prompting, statistical agent selection
- **Blind Comparative Evaluation**: Judge LLMs rank reconstructions without knowing which method produced them

### Engineering Features
- **Type-Safe Configuration**: Pydantic models ensure prompt validation and reproducibility
- **Crash Resilience**: SQLite storage with immediate writes allows resuming after failures
- **Provider Agnostic**: Works with 100+ LLM providers via LiteLLM
- **Template-Driven Prompts**: Jinja2 templates separate prompt logic from Python code
- **Provenance Tracking**: Every text segment carries complete source information

🔧 See engineering details in [`crash_resilient_sqlite_patterns.md`](crash_resilient_sqlite_patterns.md)

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────┐
│  LLM Interface (belletrist/llm.py)      │
│  Abstraction over LiteLLM              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Prompt Engineering                     │
│  PromptMaker + Pydantic + Jinja         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Data & Storage                         │
│  DataSampler + SegmentStore + Evals     │
└─────────────────────────────────────────┘
```

### Core Components (belletrist/ package)

**LLM Interface**:
- `llm.py`: Clean abstraction over LiteLLM with structured output support
  - `LLM.complete()`: Basic text completion
  - `LLM.complete_with_schema()`: Type-safe structured output with Pydantic validation
  - Graceful fallback between strict JSON schema and json_object modes

**Prompt Engineering**:
- `prompt_maker.py`: Jinja2-based template rendering with type-safe configs
- `prompts/prompt_models.py`: Pydantic models defining all prompt configurations
- `prompts/templates/*.jinja`: Actual prompt templates (separates logic from code)

**Data & Storage**:
- `data_sampler.py`: Text loading with full provenance tracking via `TextSegment` dataclass
- `segment_store.py`: SQLite catalog for curated passage examples with CRUD operations
- `style_evaluation_store.py`: Crash-resilient storage for evaluation experiments

**Pattern**: Every LLM call is saved to SQLite immediately after completion. If your experiment crashes at call #287 out of 320, you resume from #288, not #0.

## Notebooks

### 1. `style_segmentor.ipynb`
**Purpose**: Build a catalog of exemplary passages from source texts.

**Workflow**:
1. Load chapters from source files
2. LLM analyzes text to identify 5-20 exemplary passages
3. Each passage gets: craft move label, teaching note, 2-5 tags
4. Passages saved to SQLite catalog with full provenance
5. Catalog becomes searchable repository for reconstruction

**Output**: `segments.db` containing curated passages

**When to use**: Run first to build your passage catalog before evaluation experiments.

---

### 2. `style_evaluation_statistical.ipynb` ⭐ **START HERE**
**Purpose**: Evaluate how well different reconstruction methods capture style.

**Methods Tested**:
- **Generic**: Baseline with "write clearly" instructions
- **Fewshot**: Learns from 2-3 unrelated examples
- **Author**: Uses author name to invoke LLM's implicit knowledge
- **Agent Statistical**: Randomly selects 10 passages from catalog

**Workflow**:
1. **Neutralize**: Rewrite test samples in bland journalistic prose
2. **Reconstruct**: Generate M stochastic runs using each method with configured LLM
3. **Judge (Blind)**: Judge LLM ranks all 4 methods from 1-4 (anonymous labels)
4. **Aggregate**: Calculate mean rankings and win rates

**Output**:
- `style_eval_*.db`: Complete experiment data (crash-resilient)
- `style_eval_*.csv`: Rankings and statistics for analysis

**When to use**: Primary evaluation workflow. This is the main experiment notebook.

---

### 3. `style_evaluation_fewshot_sources.ipynb`
**Purpose**: Controlled experiment testing whether few-shot examples need to come from the same author.

**Comparison**:
- Few-shot with author's own texts
- Few-shot with different author's texts
- Few-shot with mixed sources

**When to use**: Investigate the source of few-shot effectiveness.

---

## Generated Documentation

The framework automatically generates detailed analysis reports:

- **`method_performance_summary.md`**: Mean ranks, win rates, and statistics for all methods across LLMs
- **`sample_XXX_run_Y_comparison.md`**: Side-by-side comparison of all reconstructions for a specific sample with judge reasoning (see `reconstruction_outputs/` for examples)
- **`crash_resilient_sqlite_patterns.md`**: Deep dive into the engineering patterns for durable LLM experiments

## Data

### Included Data

The `data/russell/` directory contains prose samples from Bertrand Russell:
- 7 text files with philosophical essays
- ~200-300 paragraphs per file
- Topics: education, knowledge, civilization, ethics

### Data Format

Text files should be:
- Plain text (`.txt`)
- Paragraph-separated (blank lines between paragraphs)
- UTF-8 encoded

### Using Your Own Data

To adapt this framework for a different author:

1. **Prepare texts**:
   ```bash
   mkdir data/your_author
   # Add .txt files with paragraph-separated prose
   ```

2. **Update notebook paths**:
   ```python
   DATA_PATH = Path("data/your_author")
   ```

3. **Build segment catalog**:
   ```python
   # In style_segmentor.ipynb
   AUTHOR_NAME = "Your Author"  # For metadata
   ```

4. **Run evaluations**:
   ```python
   # In evaluation notebooks
   AUTHOR_NAME = "Your Author"  # For author method
   ```

The framework makes no assumptions about genre, period, or language (though LLM performance may vary).

## Installation

### Requirements

- Python 3.8+
- API keys for LLM providers (OpenAI, Anthropic, Mistral, etc.)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/style-retrieval.git
cd style-retrieval

# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export MISTRAL_API_KEY="your_key"
# Any LiteLLM-supported provider works
```

### Configuration

Edit notebook cells to configure:
- **Models**: Which LLMs to use for reconstruction and judging
- **Methods**: Which reconstruction approaches to test
- **Samples**: How many test texts and reconstruction runs
- **Paths**: Where to find data and save databases

## Usage

### Quick Start

1. **Build a passage catalog** (optional but recommended):
   ```bash
   jupyter notebook style_segmentor.ipynb
   # Run all cells
   # Output: segments.db
   ```

2. **Evaluate reconstruction methods**:
   ```bash
   jupyter notebook style_evaluation_statistical.ipynb
   # Configure models and methods in early cells
   # Run all cells (crash-resilient: can resume if interrupted)
   # Output: style_eval_*.db, style_eval_*.csv
   ```

3. **Analyze results**:
   The notebook automatically displays:
   - Mean rankings by method
   - Win rates (% ranked 1st or 2nd)
   - Judge reasoning for specific samples
   - Side-by-side reconstructions

### Advanced Workflows

**Test few-shot source dependency**:
```bash
jupyter notebook style_evaluation_fewshot_sources.ipynb
```

**Generate comparison reports**:
```python
# Use the report generation script (see notebook cells at end)
# Produces markdown files with side-by-side comparisons
```

## Design Principles

### 1. Pydantic-First Configuration

All prompts use type-safe Pydantic models with validation:
```python
config = StyleNeutralizationConfig(
    text="Original text here..."
)
prompt = prompt_maker.render(config)
```

**Benefits**: Field validation, IDE autocomplete, self-documenting code, no typos in variable names.

### 2. Template-Driven Prompts

Prompt logic lives in Jinja templates (`prompts/templates/`), not Python strings:
```jinja
You are rewriting the following text in neutral style:

{{ text }}

Remove distinctive stylistic choices while preserving...
```

**Benefits**: Non-engineers can edit prompts, version control friendly, conditional logic via Jinja.

### 3. Provenance Tracking

All text segments carry full provenance via `TextSegment` dataclass:
```python
segment = sampler.get_paragraph_chunk(file_index=0, paragraph_range=slice(10, 15))
# Access: segment.text, .file_path, .paragraph_start, .paragraph_end
```

**Benefits**: Reproducibility, traceability, database-ready metadata.

### 4. Crash Resilience

All LLM responses saved to SQLite immediately with atomic transactions:
```python
# Evaluations can be resumed after crashes
if store.has_reconstruction(sample_id, run, method):
    print("✓ Already done (skipping)")
    continue  # Skip expensive LLM call

response = llm.complete(prompt)
store.save_reconstruction(...)  # Saved to disk NOW
```

**Benefits**: Lost $0.50 on crash, not $160. Resume from 75% complete, not 0%.

See [`crash_resilient_sqlite_patterns.md`](crash_resilient_sqlite_patterns.md) for implementation details.

### 5. Blind Evaluation

Judges see only anonymous labels (Text A, B, C, D) with randomized order:
```python
# Eliminates bias toward known methods
mapping = store.create_random_mapping(seed=deterministic_seed)
judge_config = StyleJudgeComparativeConfig(
    original_text=sample['original_text'],
    reconstruction_text_a=reconstructions[mapping.text_a],
    reconstruction_text_b=reconstructions[mapping.text_b],
    ...
)
```

**Benefits**: Unbiased rankings, scientifically sound, mapping stored for de-anonymization during analysis.

## Prompt Models

The framework includes 11 active prompt configurations (see `belletrist/prompts/prompt_models.py`):

**Segment Analysis**:
- `ExemplarySegmentAnalysisConfig`: Identifies exemplary passages with craft annotations

**Style Evaluation**:
- `StyleNeutralizationConfig`: Neutral rewrite preserving structure
- `StyleReconstructionGenericConfig`: Baseline reconstruction
- `StyleReconstructionFewShotConfig`: Few-shot learning
- `StyleReconstructionAuthorConfig`: Author name prompting
- `StyleJudgeComparativeConfig`: Blind 4-way ranking

**Strategic Retrieval**:
- `StrategicRetrievalPlannerConfig`: Holistic rhetorical analysis (planning phase)
- `HolisticStyledRewriteConfig`: Rewriter using strategically-selected examples
- `StatisticalFewShotRewriteConfig`: Pure few-shot with random catalog examples

Each config maps to a `.jinja` template in `prompts/templates/`.

## LLM Provider Support

Built on LiteLLM, supporting 100+ providers:
```python
# OpenAI
llm = LLM(LLMConfig(model="gpt-4o", api_key=key))

# Anthropic
llm = LLM(LLMConfig(model="claude-sonnet-4-5", api_key=key))

# Mistral
llm = LLM(LLMConfig(model="mistral/mistral-large-2411", api_key=key))

# Qwen via Together AI
llm = LLM(LLMConfig(model="together_ai/Qwen/Qwen3-235B", api_key=key))
```

All prompts are provider-agnostic. The same evaluation code works across all models.

## Evaluation Metrics

**Mean Rank**: Lower is better (1.0 = always best, 4.0 = always worst)
```
Agent Statistical:  1.74
Fewshot:           1.94
Author:            2.65
Generic:           3.67
```

**Win Rates**: Percentage ranked 1st or in top-2
```
Agent Statistical:  48% wins, 85% top-2
Fewshot:           33% wins, 78% top-2
Author:            18% wins, 31% top-2
Generic:            1% wins,  6% top-2
```

## Customization

### Adding a New Reconstruction Method

1. **Create Pydantic config** in `prompts/prompt_models.py`:
```python
class MyMethodConfig(BasePromptConfig):
    content_summary: str = Field(..., min_length=10)
    my_parameter: str = Field(...)

    @classmethod
    def template_name(cls) -> str:
        return "my_method"
```

2. **Create Jinja template** at `prompts/templates/my_method.jinja`:
```jinja
Expand this content summary into a full text:

{{ content_summary }}

Apply {{ my_parameter }} to enhance the writing...
```

3. **Add to evaluation notebook**:
```python
METHODS = ['generic', 'fewshot', 'author', 'my_method']
RECONSTRUCTORS_CFGS = {
    'my_method': MyMethodConfig
}
RECONSTRUCTORS_KWARGS = {
    'my_method': {'my_parameter': 'some value'}
}
```

The framework handles the rest (storage, judging, analysis).

### Changing Judge Criteria

Edit `prompts/templates/style_judge_comparative.jinja` to modify ranking instructions. The judge uses structured output (`StyleJudgmentComparative` Pydantic model) with schema validation.

## Project Structure

```
style-retrieval/
├── belletrist/                      # Core framework package
│   ├── llm.py                       # LLM interface (LiteLLM wrapper)
│   ├── prompt_maker.py              # Template rendering engine
│   ├── data_sampler.py              # Text sampling with provenance
│   ├── segment_store.py             # SQLite catalog for passages
│   ├── style_evaluation_store.py    # Crash-resilient evaluation storage
│   ├── agent_rewriter.py            # Agent-based reconstruction workflows
│   └── prompts/
│       ├── prompt_models.py         # Pydantic configuration models
│       └── templates/               # Jinja2 prompt templates
│           ├── style_neutralization.jinja
│           ├── style_reconstruction_*.jinja
│           └── style_judge_comparative.jinja
├── data/
│   └── russell/                     # Sample data (Bertrand Russell essays)
├── style_segmentor.ipynb            # Build passage catalog
├── style_evaluation_statistical.ipynb  # Main evaluation (⭐ start here)
├── style_evaluation_fewshot_sources.ipynb  # Few-shot source comparison
├── method_performance_summary.md    # Generated: Results summary
├── crash_resilient_sqlite_patterns.md  # Generated: Engineering guide
├── reconstruction_outputs/          # Generated: Sample comparisons
├── requirements.txt
├── CLAUDE.md                        # Internal: Instructions for Claude Code
└── README.md
```

## License

MIT License - see LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{style_retrieval_2025,
  author = {[Your Name]},
  title = {Style Retrieval and Evaluation Framework},
  year = {2025},
  url = {https://github.com/yourusername/style-retrieval},
  note = {Blog post: [URL]}
}
```

## Contributing

Contributions welcome! Areas of interest:
- Additional reconstruction methods (e.g., chain-of-thought style analysis, hybrid approaches)
- Support for poetry/dialogue/other genres
- Multilingual evaluation
- Alternative evaluation metrics (perplexity, embedding similarity)
- UI for catalog browsing and comparison viewing
- Integration with writing tools

Please open an issue to discuss major changes before submitting PRs.

## Contact

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- Blog: [Your blog URL]

---

**Credits**: This framework builds on research in computational stylistics and prompt engineering. Special thanks to the LiteLLM project for provider abstraction and to Anthropic, OpenAI, Mistral, and Moonshot AI for API access.
