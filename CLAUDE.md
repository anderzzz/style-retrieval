# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Style Retrieval** is a systematic framework for LLM and agentic workflows with structured sampling capabilities. This project builds on the architectural patterns from `russell_writes` but takes a new direction focused on systematic experimentation with prompt-based models.

Core philosophy: Pydantic-first configuration → Template-driven prompts → Systematic sampling → Type-safe validation

## Development Commands

### Installing Dependencies
```bash
pip install -r requirements.txt
```

Required environment variables:
- `MISTRAL_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` (depending on model choice)
- Any LiteLLM-supported provider API key

### Testing LLM Functionality
```bash
python belletrist/llm.py
```

## High-Level Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────┐
│  LLM Interface                          │
│  LLM/ToolLLM → LiteLLM abstraction      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Prompt Engineering                     │
│  PromptMaker + Pydantic + Jinja         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Data & Tools                           │
│  DataSampler + Tool framework           │
└─────────────────────────────────────────┘
```

### Core Components

1. **LLM Interface** (`belletrist/llm.py`): Clean abstraction over LiteLLM
   - `LLM`: Basic completion with text/JSON modes
   - `ToolLLM`: Agentic workflows with tool execution
   - Structured output via `complete_with_schema()` with Pydantic validation
   - Handles markdown-wrapped JSON automatically
   - Set `max_tokens` sufficiently high for large structured outputs

2. **Prompt Engineering** (`belletrist/prompt_maker.py`, `belletrist/prompts/`):
   - Pydantic models define template variables with validation
   - Jinja templates in `prompts/templates/` contain prompt logic
   - Type-safe, declarative prompt construction

3. **Data Sampling** (`belletrist/data_sampler.py`):
   - Load and sample text from files with full provenance
   - `TextSegment` dataclass tracks file_index, paragraph ranges
   - Supports deterministic and weighted random sampling

4. **Segment Storage** (`belletrist/segment_store.py`):
   - SQLite-backed catalog for curated few-shot examples
   - Stores passages with craft move labels and teaching notes
   - Skills pattern: browse catalog → select segments → retrieve full text
   - CRUD operations: save, get, delete, clear_all

5. **Tool Framework** (`belletrist/tools/`):
   - Abstract `Tool` base class for LLM-callable functions
   - Pydantic-based `ToolConfig` for type safety
   - OpenAI function calling schema generation

## Core Architectural Patterns

### 1. Pydantic-First Configuration

All prompts and LLM calls use type-safe Pydantic models:

```python
# Bad: string building
prompt = f"Analyze this: {text}"

# Good: type-safe config
config = BasicPromptConfig(
    role="expert analyst",
    question="What are the key themes?",
    context=text
)
prompt = prompt_maker.render(config)
```

Every `*Config` class maps to a `.jinja` template via `template_name()` classmethod.

### 2. Template-Driven Prompts

**Never build prompts with string concatenation.** Use Jinja templates in `/prompts/templates/`:

```python
# Prompt logic lives in prompts/templates/basic_prompt.jinja
# Python code is declarative:
config = BasicPromptConfig(role="...", question="...")
prompt = prompt_maker.render(config)
response = llm.complete(prompt)
```

To add a new prompt type:
1. Create `MyPromptConfig(BasePromptConfig)` in `prompts/prompt_models.py`
2. Implement `template_name()` classmethod
3. Create `my_prompt.jinja` in `prompts/templates/`
4. Export config in `prompts/__init__.py`

### 3. Structured Output with Schema Validation

For type-safe structured responses, use `complete_with_schema()`:

```python
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
    summary: str
    key_points: list[str]
    confidence: float = Field(..., ge=0, le=1)

# LLM call with automatic validation
response = llm.complete_with_schema(
    prompt="Analyze this text...",
    schema_model=AnalysisResult
)

result = response.content  # Already a validated AnalysisResult instance
print(result.summary, result.key_points)
```

The method automatically:
- Tries strict JSON schema mode (if provider supports it)
- Falls back to json_object mode with manual validation
- Strips markdown code block wrappers (````json ... ```)
- Parses JSON and validates against Pydantic model
- Returns validated model instance in `response.content`

**Important**: Set `max_tokens` sufficiently high (e.g., 16384) for large structured outputs to avoid truncation.

### 4. Provenance Tracking

All text segments carry full provenance:

```python
sampler = DataSampler("data/texts/")
segment = sampler.sample_segment(p_length=5)

# segment has: .text, .file_index, .paragraph_start, .paragraph_end, .file_path
print(f"Sampled from {segment.file_path.name}, paragraphs {segment.paragraph_start}-{segment.paragraph_end}")
```

### 5. Agentic Workflows with Tools

`ToolLLM` enables multi-turn tool execution:

```python
from belletrist.tools import Tool, ToolConfig

# Define a custom tool
class MyTool(Tool):
    def execute(self, **kwargs) -> str:
        # Tool logic here
        return "result"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.config.name,
                "description": self.config.description,
                "parameters": {...}
            }
        }

# Register and use
llm = ToolLLM(LLMConfig(model="gpt-4", api_key=key))
llm.register_tool(MyTool(ToolConfig(name="my_tool", description="...")))
response = llm.complete_with_tools("Use the tool to solve this problem")
```

### 6. Style Retrieval Workflow

**Purpose**: Build a catalog of exemplary passages with craft annotations for few-shot learning.

The workflow extracts passages from source texts and stores them with teachable craft descriptions:

```python
from belletrist import DataSampler, SegmentStore, LLM, LLMConfig, PromptMaker
from belletrist.prompts import ExemplarySegmentAnalysisConfig, ExemplarySegmentAnalysis

# Load chapter
sampler = DataSampler("data/texts/")
chapter = sampler.get_paragraph_chunk(file_index=0, paragraph_range=slice(0, 50))

# Configure analysis prompt
config = ExemplarySegmentAnalysisConfig(
    chapter_text=chapter.text,
    file_name=chapter.file_path.name,
    num_segments=12  # Request 12 passages
)
prompt = prompt_maker.render(config)

# Get structured analysis
llm = LLM(LLMConfig(model="gpt-4o", api_key=key, max_tokens=16384))
response = llm.complete_with_schema(prompt, schema_model=ExemplarySegmentAnalysis)
analysis = response.content  # ExemplarySegmentAnalysis with passages

# Store passages in catalog
with SegmentStore("segments.db") as store:
    for passage in analysis.passages:
        # Locate passage in chapter (text matching, not relying on LLM numeracy)
        para_range = find_passage_in_chapter(passage.text, chapter.text, sampler, ...)
        text_segment = sampler.get_paragraph_chunk(file_index, slice(*para_range))

        # Save with craft annotations
        segment_id = store.save_segment(
            text_segment=text_segment,
            craft_move=passage.craft_move,
            teaching_note=passage.teaching_note,
            tags=passage.tags
        )
```

**Key models**:
- **`ExemplarySegmentAnalysisConfig`**: Prompt config with `chapter_text`, `file_name`, `num_segments`
- **`ExemplarySegment`**: Single passage with `text`, `craft_move`, `teaching_note`, `tags`
- **`ExemplarySegmentAnalysis`**: Complete analysis with `passages` list and `overall_observations`

**Segment catalog schema**:
```sql
CREATE TABLE segments (
    segment_id TEXT PRIMARY KEY,      -- seg_001, seg_002, ...
    file_index INTEGER,               -- Source file index
    paragraph_start INTEGER,          -- Paragraph range (provenance)
    paragraph_end INTEGER,
    file_name TEXT,
    text TEXT,                        -- Full passage text
    craft_move TEXT,                  -- e.g., "concessive_pivot"
    teaching_note TEXT,               -- "Notice how the author..."
    tags TEXT,                        -- JSON array of tags
    created_at TIMESTAMP,
    source TEXT                       -- "llm_analysis"
);
```

**SegmentStore API**:
```python
with SegmentStore("segments.db") as store:
    # Save passage
    seg_id = store.save_segment(text_segment, craft_move, teaching_note, tags)

    # Retrieve by ID
    record = store.get_segment("seg_001")
    print(record.craft_move, record.teaching_note, record.text)

    # Browse catalog (skills pattern)
    catalog = store.browse_catalog(limit=10)
    for entry in catalog:
        print(f"{entry['segment_id']}: {entry['craft_move']}")

    # Search by tag
    segments = store.search_by_tag("concessive_structure")

    # List all tags with counts
    tags = store.list_all_tags()  # {'concessive_structure': 8, ...}

    # Delete operations
    store.delete_segment("seg_003")       # Delete one
    count = store.clear_all(confirm=True) # Delete all (safety check)
    total = store.get_count()             # Total segments
```

**Prompt design principles** (`exemplary_segment_analysis.jinja`):
- **Author anonymity**: Prompt explicitly forbids mentioning author names
- **Parameterized requests**: `{{ num_segments }}` makes count configurable
- **Craft focus**: Emphasizes teachable techniques, not content analysis
- **Text-based output**: LLM returns full passage text (not paragraph numbers)

**Why text-based extraction?**
- LLMs are poor at counting paragraphs accurately
- Full text extraction is more reliable
- Provenance recovered via text matching against source

## File Organization

### `/belletrist/`

Core module structure:

- **`llm.py`**: LLM wrappers (LLM, ToolLLM, LLMConfig, Message, LLMResponse)
- **`prompt_maker.py`**: Template rendering engine
- **`data_sampler.py`**: Text loading and sampling with provenance
- **`segment_store.py`**: SQLite-backed catalog for curated passage examples
- **`prompts/prompt_models.py`**: All Pydantic prompt configurations
- **`prompts/templates/`**: Jinja template files (*.jinja)
- **`tools/tool.py`**: Abstract Tool base classes

### Naming Conventions

- **Config classes**: `*Config` suffix (e.g., `BasicPromptConfig`)
- **Templates**: Snake case matching `template_name()` (e.g., `basic_prompt.jinja`)
- **Models**: Pydantic models with field validation

### Module Exports

Public API exposed via `/belletrist/__init__.py`:
```python
from belletrist import (
    # LLM interface
    LLM, ToolLLM, LLMConfig, Message, LLMResponse, LLMRole,
    # Core utilities
    PromptMaker, DataSampler, TextSegment,
    # Segment catalog
    SegmentStore, SegmentRecord
)

# Prompt configs imported separately
from belletrist.prompts import (
    BasePromptConfig, BasicPromptConfig,
    ExemplarySegmentAnalysisConfig, ExemplarySegmentAnalysis
)
```

## Inherited Prompt Models

This project inherits extensive prompt models from `russell_writes`:

### Foundational Models
- `BasicPromptConfig`: Generic role-based prompts
- `PreambleTextConfig`: Text container for analysis
- `PreambleInstructionConfig`: Static instructional preamble

### Specialist Analyst Models (from literary analysis)
- `SyntacticianConfig`: Syntax and sentence structure
- `LexicologistConfig`: Vocabulary and diction
- `InformationArchitectConfig`: Information structure
- `RhetoricianConfig`: Rhetorical strategy
- `EfficiencyAuditorConfig`: Economy and compression

### Integration & Synthesis Models
- `CrossPerspectiveIntegratorConfig`: Multi-analyst integration
- `CrossTextSynthesizerConfig`: Cross-text pattern extraction
- `SynthesizerOfPrinciplesConfig`: Prescriptive guide generation

### Style Evaluation Models
- `StyleFlatteningConfig`: Content extraction
- `StyleFlatteningAggressiveConfig`: Aggressive compression
- `StyleReconstructionGenericConfig`: Baseline reconstruction
- `StyleReconstructionFewShotConfig`: Few-shot learning
- `StyleReconstructionAuthorConfig`: Author name prompting
- `StyleReconstructionInstructionsConfig`: Instruction-based reconstruction
- `StyleJudgeComparativeConfig`: Blind 4-way ranking
- `StyleJudgeComparative5WayConfig`: Blind 5-way ranking

### Style Retrieval Models (Active)
- **`ExemplarySegmentAnalysisConfig`**: Identifies exemplary passages demonstrating craft moves
  - Has active template: `exemplary_segment_analysis.jinja`
  - Parameters: `chapter_text`, `file_name`, `num_segments` (5-20)
  - Returns: `ExemplarySegmentAnalysis` with passages and observations
- **`ExemplarySegment`**: Single passage with craft annotations
  - Fields: `text`, `craft_move`, `teaching_note`, `tags`
- **`ExemplarySegmentAnalysis`**: Complete analysis result
  - Fields: `passages` (list), `overall_observations` (optional)

**Note**: Most inherited models lack corresponding `.jinja` templates. Template creation follows actual use cases. The style retrieval models above are fully implemented with working templates.

## LiteLLM Provider Flexibility

The `LLM` class wraps LiteLLM, supporting 100+ providers:

```python
# OpenAI
llm = LLM(LLMConfig(model="gpt-4o", api_key=os.environ['OPENAI_API_KEY']))

# Mistral
llm = LLM(LLMConfig(model="mistral/mistral-large-2411", api_key=os.environ['MISTRAL_API_KEY']))

# Anthropic
llm = LLM(LLMConfig(model="claude-3-5-sonnet-20241022", api_key=os.environ['ANTHROPIC_API_KEY']))
```

All prompts are provider-agnostic.

## Common Workflows

### Creating a New Prompt Type

1. **Define config** in `prompts/prompt_models.py`:
```python
class MyAnalysisConfig(BasePromptConfig):
    text: str = Field(..., min_length=1)
    focus_area: str = Field(...)

    @classmethod
    def template_name(cls) -> str:
        return "my_analysis"
```

2. **Create template** at `prompts/templates/my_analysis.jinja`:
```jinja
You are analyzing the following text with focus on {{ focus_area }}.

Text:
{{ text }}

Provide a detailed analysis.
```

3. **Export** in `prompts/__init__.py`:
```python
from belletrist.prompts.prompt_models import MyAnalysisConfig
```

4. **Use**:
```python
from belletrist import PromptMaker, LLM, LLMConfig
from belletrist.prompts import MyAnalysisConfig

maker = PromptMaker()
llm = LLM(LLMConfig(model="gpt-4o", api_key=key))

config = MyAnalysisConfig(text="...", focus_area="themes")
prompt = maker.render(config)
response = llm.complete(prompt)
```

### Systematic Sampling Workflow

```python
from belletrist import DataSampler

# Load corpus
sampler = DataSampler("data/texts/")

# Sample with provenance
segment = sampler.sample_segment(p_length=5)
print(f"Sampled: {segment.file_path.name} [{segment.paragraph_start}:{segment.paragraph_end}]")

# Deterministic retrieval
segment = sampler.get_paragraph_chunk(file_index=0, paragraph_range=slice(10, 15))

# Iterate through chunks
for chunk in sampler.iter_paragraph_chunks(file_index=0, chunk_size=3):
    print(f"Processing: {chunk.text[:100]}...")
```

### Running the Style Retrieval Script

The main workflow is implemented in `runs/style_retrieval.py`:

```bash
# Configure parameters in the script first (API key, model, data paths)
python runs/style_retrieval.py
```

**Configuration parameters** (edit in script):
```python
# API Configuration
API_KEY = os.environ.get('TOGETHER_AI_API_KEY', '')
MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507"

# Data Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "russell"
DB_PATH = Path(__file__).parent.parent / "segments.db"

# Analysis Parameters
FILE_INDEX = 0           # Which file to analyze
CHAPTER_START = 9        # Starting paragraph
CHAPTER_END = 41         # Ending paragraph
TEMPERATURE = 0.7        # LLM temperature
```

**Workflow phases**:
1. **Analysis**: LLM identifies exemplary passages from chapter
2. **Storage**: Passages saved to SQLite catalog with craft annotations
3. **Retrieval Demo**: Browse catalog, search by tags, retrieve segments

**Output**: `segments.db` containing catalog of passages ready for agent retrieval.

## Design Rationale

### Why Pydantic-First?

- **Type safety**: Catch configuration errors at validation time
- **Documentation**: Field descriptions serve as inline docs
- **IDE support**: Autocomplete and type hints
- **Validation**: Automatic range checks, string length, required fields

### Why Template-First Prompts?

Alternatives considered:
- **String concatenation**: Hard to maintain, no validation
- **Python-only**: Harder for non-coders to iterate
- **JSON configs**: Less expressive than templates

Jinja provides:
- Conditional sections (`{% if include_X %}`)
- Loops (`{% for item in items %}`)
- Includes (`{% include 'subtemplate.jinja' %}`)
- Clear separation: prompt logic in `.jinja`, validation in Python

### Why LiteLLM?

- **Provider flexibility**: Switch models without code changes
- **Unified interface**: Same code works with OpenAI, Anthropic, Mistral, etc.
- **Cost optimization**: Easy to experiment with cheaper models
- **Feature support**: JSON mode, tools, streaming across providers

## Project Direction

This is a clean break from `russell_writes` to focus on:

1. **Few-shot catalog building**: Extract exemplary passages with craft annotations for agentic writing
2. **Agentic workflows**: Enable agents to browse catalogs, select relevant examples, and apply techniques
3. **Skills pattern**: Agents autonomously retrieve context via browsable catalogs (not pre-loaded prompts)
4. **Type safety**: Maintain strict Pydantic validation throughout
5. **Author anonymity**: Test on private texts without revealing authorship

### Current Focus: Style Retrieval for Writing Agents

The active workflow builds catalogs of passages annotated with craft moves:
- **Input**: Chapters from source texts (private or public)
- **Processing**: LLM identifies exemplary passages demonstrating teachable techniques
- **Storage**: SQLite catalog with passages, craft moves, teaching notes, tags
- **Retrieval**: Agents browse catalog by tags, select relevant examples, retrieve full text
- **Application**: Agents use retrieved examples as few-shot demonstrations for writing tasks

This enables writing agents to dynamically select appropriate stylistic examples rather than relying on static prompts.

## Known Considerations

- **Template Creation**: Most inherited prompt models lack corresponding `.jinja` templates. Create templates only when actively using a model. The `exemplary_segment_analysis.jinja` template is fully implemented and tested.

- **API Keys**: Set environment variables for your chosen provider before running any LLM code.

- **Token Limits**: For large structured outputs, set `max_tokens` sufficiently high (e.g., 16384). The default provider limit may truncate JSON responses. Monitor for `JSONDecodeError` indicating truncation.

- **Token Costs**: Use cheaper models for experimentation. Consider `temperature=0` for deterministic results. Style retrieval analysis can be expensive with large chapter inputs.

- **Error Handling**: `complete_with_schema()` raises `ValueError` if JSON parsing or validation fails. Always handle these exceptions in production code.

- **Database Schema**: The `SegmentStore` schema uses `craft_move` and `teaching_note` fields (not `functional_description`/`formal_description`). If you have an old database, delete it and regenerate with the new schema.

- **Passage Location**: The workflow locates passages via text matching (not paragraph indices from LLM). This is more reliable but requires exact text extraction from the LLM. If passage location fails, it falls back to approximate ranges with warnings.

- **Author Anonymity**: The prompt explicitly forbids mentioning author names to enable testing on private texts. This is enforced via prompt design, not post-processing.
