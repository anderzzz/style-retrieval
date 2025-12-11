"""
Style Rewrite: Multi-agent workflow for styled text generation.

This script demonstrates:
1. Planning Agent - analyzes flattened text, creates paragraph-level plan
2. Python Retrieval - searches catalog for matching craft examples (heuristic-based)
3. Rewriting Agent - uses examples to generate styled output

Follows Option B architecture: deterministic retrieval with LLM planning + rewriting.

USAGE:
    1. Set configuration parameters below (API key, model, paths, input text)
    2. Ensure segments.db exists (run style_retrieval.py first to build catalog)
    3. Run: python runs/style_rewrite.py
    4. Monitor output as workflow progresses
"""
import os
from pathlib import Path

from belletrist import LLM, LLMConfig, PromptMaker, DataSampler, SegmentStore
from belletrist.agent_rewriter import agent_rewrite


# ============================================================================
# CONFIGURATION - Modify these parameters before running
# ============================================================================

# API Configuration
#API_KEY = os.environ.get('MISTRAL_API_KEY', '')
API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
#API_KEY = os.environ.get('TOGETHER_AI_API_KEY', '')  # or set directly: "sk-..."
#MODEL = "mistral/mistral-large-2512"
MODEL = 'anthropic/claude-sonnet-4-5-20250929'
#MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput"

# Alternative examples:
# MODEL = "gpt-4o"
#
# API_KEY = os.environ.get('TOGETHER_AI_API_KEY', '')
# MODEL = "together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507"

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "russell"
DB_PATH = Path(__file__).parent.parent / "segments_mistral.db"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs"
OUTPUT_FILE_NAME = "rewrite_output_sonnet"

# LLM Parameters
PLANNING_TEMPERATURE = 0.5       # Lower for deterministic planning
REWRITING_TEMPERATURE = 0.7      # Moderate for creative rewriting

MAX_TOKENS_PLANNING = 8192   # Large JSON plans need generous token budget
MAX_TOKENS_REWRITING = 8192  # Full paragraph rewrites need generous token budget

# Workflow Parameters
NUM_EXAMPLES_PER_PARAGRAPH = 3
TARGET_STYLE = "balanced, lucid, rhythmically varied with concessive structure"
CREATIVE_LATITUDE = "conservative"  # "conservative", "moderate", or "aggressive"

# Input Text (flattened/style-sparse)
#INPUT_TEXT = """
#Democracy has flaws. It can be inefficient and messy. Leaders are elected by popularity rather than competence. The process is slow.
#
#But democracy remains the best system. It provides accountability. Citizens can remove bad leaders. No other system has proven superior.
#
#The alternatives are worse. Autocracy lacks checks on power. Technocracy ignores popular will. Aristocracy entrenches privilege.
#
#Democracy's strength is its weakness. Slowness prevents rash action. Messiness reflects genuine debate. Inefficiency protects against tyranny.
#"""
INPUT_TEXT = """
- Educational theory has changed significantly since earlier times, particularly in relation to democracy and accessibility.

- Before the nineteenth century, the two major figures in educational reform were Locke and Rousseau.

- Both Locke and Rousseau rejected many widespread educational errors of their time and contributed meaningfully to educational thought.

- Despite their progressive reputations, neither Locke nor Rousseau extended their ideas to support universal education.

- Both thinkers focused on the education of a single aristocratic boy, assuming that one adult could devote full time to one child.

- This model, while potentially effective, is not scalable because it is impossible for every child to have a full-time tutor.

- Therefore, the model is limited to a privileged minority and cannot exist in a just society.

- Modern educational thought requires that any acceptable system must be applicable to all children, at least in theory.

- The ideal educational system must be democratic, even if full realization of this ideal is not immediately possible.

- This democratic ideal is now widely accepted.

- The author supports keeping democracy as a guiding principle in educational reform.

- Any educational method advocated should be capable of universal application.

- Individuals may currently use non-universal methods if better than available public options, but such choices should not be considered ideal.

- The democratic principle, even in a minimal form, is absent in the works of both Locke and Rousseau.

- Rousseau, despite opposing aristocracy, failed to apply his anti-aristocratic views to education.

- Clarity is essential when discussing democracy and education.

- Insisting on uniformity in education would be harmful.

- There are natural differences in ability among children: some are more capable of benefiting from advanced education.

- Teachers also vary in quality, and it is impossible for all students to be taught by the best teachers.

- Even if higher education were suitable for everyone, it is not currently feasible for all to access it.

- A rigid application of democratic equality might lead to the conclusion that no one should receive higher education, which would be detrimental.

- Such an outcome would hinder scientific progress and lower the overall educational standard in the future.

- Progress should not be sacrificed for mechanical equality.

- Educational democracy must be pursued carefully to preserve existing valuable educational achievements, even if they originated in unjust social conditions.

- An educational method cannot be considered satisfactory if it cannot, in principle, be made universal.

- Children of wealthy families often receive disproportionate attention from multiple caregivers, such as a mother, nurse, nurserymaid, and domestic servants.

- This level of individual attention cannot be provided to all children in any feasible social system.

- It is uncertain whether such careful, dependent upbringing benefits children, and it may foster parasitic behavior.

- No impartial observer can justify special educational advantages for the few, except in cases of exceptional need or ability, such as genius or intellectual disability.

- At present, wise parents may choose non-universal educational methods for their children, especially for experimental purposes.

- Such experimental methods should be ones that could, in principle, be extended to all if proven effective.

- Methods that are inherently limited to a privileged few should not be promoted.

- Some of the most valuable developments in modern education originated from democratic contexts.

- For example, Montessori’s work began in nursery schools in impoverished urban areas.

- In higher education, exceptional opportunities should be available for those with exceptional ability.

- Outside of cases involving exceptional talent, there is no justification for educational systems that cannot be universally adopted.

- A second modern trend in education is the shift from ornamental to useful education, which is related to democratic values but more debatable.

- Veblen’s “Theory of the Leisure Class” discusses the link between ornamentation and aristocracy, though only the educational implications are relevant here.

- In male education, this trend appears in the debate between classical education (ornamental) and modern, science-based education (useful).

- In female education, the conflict manifests as a choice between the ideal of the “gentlewoman” and training for self-sufficiency.

- The educational issues concerning women have been complicated by the pursuit of sex equality.

- Women’s education has often aimed to replicate the education given to boys, even when that education is not inherently valuable.

- Women educators have sometimes emphasized the same “useless” knowledge given to boys and have resisted including training related to motherhood.

- These conflicting goals make the shift from ornamental to useful education less clear-cut in the context of female education.

- The decline of the “fine lady” ideal is a significant example of the broader trend toward useful education.

- To clarify the argument, the author will focus only on male education for the remainder of the discussion.

- Several educational controversies are partially shaped by the distinction between useful and ornamental education.

- Examples include: whether boys should study classics or science, with classics seen as ornamental and science as useful.

- Another issue is whether education should quickly become technical training for a specific trade or profession, where usefulness is a central concern.

- The debate over whether children should be taught correct speech and polite manners involves questioning whether these are merely aristocratic relics.

- The value of art appreciation is questioned, particularly for those who are not artists.

- The proposal for phonetic spelling is another issue influenced by the useful vs. ornamental framework.

- Many educational debates are, at least in part, framed by the underlying conflict between usefulness and ornamentation.
"""

# ============================================================================


def main():
    """Main workflow: plan, retrieve, rewrite."""

    # Validate configuration
    print("="*60)
    print("STYLE REWRITE WORKFLOW")
    print("="*60)
    print("\n[Setup] Validating configuration...")

    if not API_KEY:
        raise ValueError(
            "API_KEY not set. Please configure API_KEY in the configuration section."
        )

    if not DB_PATH.exists():
        raise ValueError(
            f"Segment database not found: {DB_PATH}\n"
            f"Please run style_retrieval.py first to build the catalog."
        )

    print(f"        Model: {MODEL}")
    print(f"        Database: {DB_PATH}")
    print(f"        Target Style: {TARGET_STYLE}")
    print(f"        Creative Latitude: {CREATIVE_LATITUDE}")

    # Initialize components
    print("\n[Setup] Initializing components...")

    sampler = DataSampler(DATA_PATH)
    print(f"        ✓ DataSampler loaded {len(sampler.fps)} files")

    # Separate LLMs for planning and rewriting (different temperatures)
    planning_llm = LLM(LLMConfig(
        model=MODEL,
        api_key=API_KEY,
        temperature=PLANNING_TEMPERATURE,
        max_tokens=MAX_TOKENS_PLANNING
    ))
    print(f"        ✓ Planning LLM configured (temp={PLANNING_TEMPERATURE})")

    rewriting_llm = LLM(LLMConfig(
        model=MODEL,
        api_key=API_KEY,
        temperature=REWRITING_TEMPERATURE,
        max_tokens=MAX_TOKENS_REWRITING
    ))
    print(f"        ✓ Rewriting LLM configured (temp={REWRITING_TEMPERATURE})")

    prompt_maker = PromptMaker()
    print(f"        ✓ PromptMaker ready")

    # Validate segment database exists
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Segment database not found: {DB_PATH}\n"
            f"Please run style_retrieval.py first to build the catalog."
        )
    print(f"\n[Setup] Segment database validated: {DB_PATH}")

    # ====================================================================
    # PHASES 1-3: PLANNING → RETRIEVAL → REWRITING
    # ====================================================================
    print("\n" + "="*60)
    print("AGENT-BASED REWRITING WORKFLOW")
    print("="*60)
    print("\nPhases:")
    print("  Planning: Analyzing text structure...")
    print("  Retrieval: Searching catalog for examples...")
    print("  Rewriting: Generating styled output...")
    print()

    print(f"Input text length: {len(INPUT_TEXT)} characters")
    print(f"Input preview: {INPUT_TEXT[:150]}...\n")

    # Call the agent_rewrite wrapper (silent operation)
    with SegmentStore(DB_PATH) as store:
        styled_text = agent_rewrite(
            flattened_content=INPUT_TEXT,
            segment_store=store,
            planning_llm=planning_llm,
            rewriting_llm=rewriting_llm,
            prompt_maker=prompt_maker,
            creative_latitude=CREATIVE_LATITUDE,
            num_examples_per_paragraph=NUM_EXAMPLES_PER_PARAGRAPH
        )

    print("✓ All phases complete!")

    # ====================================================================
    # OUTPUT
    # ====================================================================
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)

    print("\n" + "─"*60)
    print("ORIGINAL TEXT:")
    print("─"*60)
    print(INPUT_TEXT)

    print("\n" + "─"*60)
    print("STYLED OUTPUT:")
    print("─"*60)
    print(styled_text)
    print("─"*60)

    # Save output
    OUTPUT_PATH.mkdir(exist_ok=True)
    output_file = OUTPUT_PATH / f"{OUTPUT_FILE_NAME}.txt"
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("ORIGINAL TEXT\n")
        f.write("="*60 + "\n\n")
        f.write(INPUT_TEXT)
        f.write("\n\n")
        f.write("="*60 + "\n")
        f.write("STYLED OUTPUT\n")
        f.write("="*60 + "\n\n")
        f.write(styled_text)

    print(f"\n✓ Output saved to: {output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
