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

2. **Prompt Engineering** (`belletrist/prompt_maker.py`, `belletrist/prompts/`):
   - Pydantic models define template variables with validation
   - Jinja templates in `prompts/templates/` contain prompt logic
   - Type-safe, declarative prompt construction

3. **Data Sampling** (`belletrist/data_sampler.py`):
   - Load and sample text from files with full provenance
   - `TextSegment` dataclass tracks file_index, paragraph ranges
   - Supports deterministic and weighted random sampling

4. **Tool Framework** (`belletrist/tools/`):
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
- Parses JSON and validates against Pydantic model
- Returns validated model instance in `response.content`

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

## File Organization

### `/belletrist/`

Core module structure:

- **`llm.py`**: LLM wrappers (LLM, ToolLLM, LLMConfig, Message, LLMResponse)
- **`prompt_maker.py`**: Template rendering engine
- **`data_sampler.py`**: Text loading and sampling with provenance
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
    LLM, ToolLLM, LLMConfig,
    PromptMaker, DataSampler,
    BasePromptConfig, BasicPromptConfig
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

**Note**: While these models are defined, most corresponding Jinja templates are not yet created. Template creation should follow actual use cases.

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

1. **Systematic experimentation**: Build frameworks for testing prompt variations
2. **Agentic workflows**: Leverage ToolLLM for multi-step reasoning
3. **Reusable patterns**: Extract generalizable prompt engineering patterns
4. **Type safety**: Maintain strict Pydantic validation throughout

The inherited prompt models provide a foundation, but the project will evolve based on specific experimental needs rather than literary analysis.

## Known Considerations

- **Template Creation**: Most inherited prompt models lack corresponding `.jinja` templates. Create templates only when actively using a model.

- **API Keys**: Set environment variables for your chosen provider before running any LLM code.

- **Token Costs**: Use cheaper models for experimentation. Consider `temperature=0` for deterministic results.

- **Error Handling**: `complete_with_schema()` raises `ValueError` if JSON parsing or validation fails. Always handle these exceptions in production code.
