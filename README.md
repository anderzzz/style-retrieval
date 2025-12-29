# Style Retrieval and Evaluation Framework

A systematic framework for analyzing, retrieving, and reconstructing literary style using Large Language Models (LLMs). This project provides tools for extracting exemplary passages from texts, building searchable catalogs of stylistic techniques, and evaluating how well different prompting methods can reconstruct an author's distinctive voice.

## Overview

The framework addresses a core challenge in computational stylistics: **Can LLMs capture and reproduce an author's distinctive style?** To answer this, we:

1. **Extract exemplary passages** from source texts, annotated with craft moves and teaching notes
2. **Build searchable catalogs** of these passages, tagged for retrieval
3. **Evaluate reconstruction methods** by comparing how well different prompting strategies can recreate the original style from neutralized content
4. **Compare LLM performance** across different models and prompting approaches

The included experiments use prose by Bertrand Russell as test data, but the framework is designed to work with any author's writing.

## Key Features

- **Segment Analysis**: LLM-powered identification of exemplary passages demonstrating teachable craft moves
- **SQLite-backed Catalog**: Persistent storage of passages with craft annotations and tags
- **Style Neutralization**: Rewrites texts in bland journalistic prose while preserving argumentative structure
- **Multiple Reconstruction Methods**: Generic baseline, few-shot learning, author name prompting, statistical agent selection
- **Blind Comparative Evaluation**: Judge LLMs rank reconstructions without knowing which method produced them
- **Cross-Model Comparison**: Bradley-Terry analysis of reconstruction quality across different LLMs
- **Type-Safe Configuration**: Pydantic models ensure prompt validation and reproducibility

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

### Core Components

- **`belletrist/llm.py`**: Clean abstraction over LiteLLM supporting 100+ providers
- **`belletrist/prompt_maker.py`**: Jinja2-based template rendering with type-safe configs
- **`belletrist/prompts/`**: Pydantic models and Jinja templates for all prompts
- **`belletrist/data_sampler.py`**: Text loading with full provenance tracking
- **`belletrist/segment_store.py`**: SQLite catalog for curated passage examples
- **`belletrist/style_evaluation_store.py`**: SQLite storage for evaluation experiments
- **`belletrist/cross_model_comparison_store.py`**: Storage for cross-model comparisons

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

### 2. `style_evaluation_statistical.ipynb`
**Purpose**: Evaluate how well different reconstruction methods capture style.

**Methods Tested**:
- **Generic**: Baseline with "write clearly" instructions
- **Few-shot**: Learns from 2-3 unrelated examples
- **Author**: Uses author name to invoke LLM's implicit knowledge
- **Agent Statistical**: Selects 10 passages statistically from catalog

**Workflow**:
1. **Neutralize**: Rewrite test samples in bland journalistic prose
2. **Reconstruct**: Generate M stochastic runs using each method
3. **Judge (Blind)**: Judge LLM ranks all 4 methods from 1-4 (anonymous)
4. **Aggregate**: Calculate mean rankings and win rates

**Output**:
- `style_eval_*.db`: Complete experiment data
- `style_eval_*.csv`: Rankings and statistics

**When to use**: Primary evaluation workflow. Run after building segment catalog.

---

### 3. `style_evaluation.ipynb`
**Purpose**: Similar to statistical evaluation but includes the **agent_holistic** method.

**Additional Method**:
- **Agent Holistic**: Strategic planning phase selects 8-10 examples based on rhetorical needs

**When to use**: Compare holistic agent approach against other methods.

---

### 4. `style_evaluation_fewshot_sources.ipynb`
**Purpose**: Controlled experiment testing whether few-shot examples need to come from the same author.

**Comparison**:
- Few-shot with author's own texts
- Few-shot with different author's texts
- Few-shot with mixed sources

**When to use**: Investigate the source of few-shot effectiveness.

---

### 5. `cross_model_evaluation.ipynb`
**Purpose**: Compare reconstruction quality across different LLMs and methods.

**Workflow**:
1. Pull reconstructions from multiple existing evaluation databases
2. Create random 4-way comparisons (e.g., Mistral+fewshot vs GPT+author vs Qwen+agent)
3. Judge ranks all 4 anonymously
4. Bradley-Terry model estimates strength of each LLM×method combination

**Output**:
- `cross_eval_*.csv`: Full judgment data
- `cross_eval_*.bt.csv`: Pairwise preferences for Bradley-Terry
- `cross_eval_*.stats.csv`: Mean ranks, win rates, confidence intervals

**When to use**: After running evaluations with multiple LLMs, aggregate results for cross-model comparison.

---

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
```

### Configuration

Edit notebook cells to configure:
- **Models**: Which LLMs to use for reconstruction and judging
- **Methods**: Which reconstruction approaches to test
- **Samples**: How many test texts and reconstruction runs
- **Paths**: Where to find data and save databases

## Usage

### Quick Start

1. **Build a passage catalog**:
   ```bash
   jupyter notebook style_segmentor.ipynb
   # Run all cells
   # Output: segments.db
   ```

2. **Evaluate reconstruction methods**:
   ```bash
   jupyter notebook style_evaluation_statistical.ipynb
   # Configure models and methods
   # Run all cells
   # Output: style_eval_*.db, style_eval_*.csv
   ```

3. **Analyze results**:
   ```python
   # In notebook, view:
   # - Mean rankings by method
   # - Win rates (% ranked 1st or 2nd)
   # - Judge reasoning
   # - Side-by-side reconstructions
   ```

### Advanced Workflows

**Compare across LLMs**:
```bash
# Run style_evaluation_statistical.ipynb with different models
# Then run cross_model_evaluation.ipynb to aggregate
```

**Test few-shot source dependency**:
```bash
jupyter notebook style_evaluation_fewshot_sources.ipynb
```

**Debug reconstruction quality**:
```python
# In evaluation notebooks, use inspection cells:
# - View judge reasoning
# - Compare reconstructions side-by-side
# - Examine prompts used
```

## Design Principles

### 1. Pydantic-First Configuration

All prompts use type-safe Pydantic models:
```python
config = StyleNeutralizationConfig(text="...")
prompt = prompt_maker.render(config)
```

### 2. Template-Driven Prompts

Prompt logic lives in Jinja templates (`prompts/templates/`), not Python:
```jinja
You are rewriting the following text in neutral style:

{{ text }}

Remove distinctive stylistic choices while preserving...
```

### 3. Provenance Tracking

All text segments carry full provenance:
```python
segment = sampler.get_paragraph_chunk(file_index=0, paragraph_range=slice(10, 15))
# segment.file_path, .paragraph_start, .paragraph_end
```

### 4. Crash Resilience

All LLM responses saved to SQLite immediately:
```python
# Evaluations can be resumed after crashes
if store.has_reconstruction(sample_id, run, method):
    print("✓ Already done (skipping)")
    continue
```

### 5. Blind Evaluation

Judges see only anonymous labels (Text A, B, C, D):
```python
# Eliminates bias toward known methods
mapping = store.create_random_mapping()
judge_config = StyleJudgeComparativeConfig(
    original_text=sample['original_text'],
    reconstruction_text_a=reconstructions[mapping.text_a],
    ...
)
```

## Prompt Models

The framework includes 11 active prompt configurations (see `belletrist/prompts/prompt_models.py`):

**Segment Analysis**:
- `ExemplarySegmentAnalysisConfig`: Identifies exemplary passages

**Style Evaluation**:
- `StyleFlatteningConfig`: Extracts content (70-90% original length)
- `StyleNeutralizationConfig`: Neutral rewrite (80-100% length, preserves structure)
- `StyleReconstructionGenericConfig`: Baseline reconstruction
- `StyleReconstructionFewShotConfig`: Few-shot learning
- `StyleReconstructionAuthorConfig`: Author name prompting
- `StyleJudgeComparativeConfig`: Blind 4-way ranking

**Strategic Retrieval**:
- `StrategicRetrievalPlannerConfig`: Holistic rhetorical analysis
- `HolisticStyledRewriteConfig`: Lean rewriter with strategic examples
- `StatisticalFewShotRewriteConfig`: Pure few-shot with catalog examples

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
```

All prompts are provider-agnostic.

## Evaluation Metrics

**Rankings**: Mean rank (1 = best, 4 = worst)
```
fewshot:            1.30
agent_statistical:  2.10
author:             2.80
generic:            3.70
```

**Win Rates**: Percentage ranked 1st or 2nd
```
fewshot:            95%
agent_statistical:  65%
author:             35%
generic:            5%
```

**Bradley-Terry Strength**: Maximum likelihood estimate of "ability"
```python
# Used in cross-model evaluation
# Higher = stronger performance
```

## Customization

### Adding a New Reconstruction Method

1. **Create Pydantic config** in `prompts/prompt_models.py`:
```python
class MyMethodConfig(BasePromptConfig):
    content_summary: str = Field(...)
    my_parameter: str = Field(...)

    @classmethod
    def template_name(cls) -> str:
        return "my_method"
```

2. **Create Jinja template** at `prompts/templates/my_method.jinja`:
```jinja
Expand this content summary:

{{ content_summary }}

Using {{ my_parameter }}...
```

3. **Add to evaluation**:
```python
METHODS = ['generic', 'fewshot', 'author', 'my_method']
RECONSTRUCTORS_CFGS = {
    'my_method': MyMethodConfig
}
RECONSTRUCTORS_KWARGS = {
    'my_method': {'my_parameter': 'some value'}
}
```

### Changing Judge Criteria

Edit `prompts/templates/style_judge_comparative.jinja` to modify ranking instructions.

### Using Different Flattening Strategies

Switch between:
- `StyleFlatteningConfig`: Content extraction (shorter)
- `StyleNeutralizationConfig`: Neutral rewrite (full-length, preserves structure)

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation if publishing]
```

## Contributing

Contributions welcome! Areas of interest:
- Additional reconstruction methods
- Support for poetry/dialogue/other genres
- Multilingual evaluation
- Alternative evaluation metrics
- UI for catalog browsing

## Contact

[Add your contact information]
