"""
Canonical craft move tags (Tier 1).

These universal tags should be reused across all authors/projects.
They describe teachable techniques that recur in many writing styles.

This module provides the single source of truth for Tier 1 tags,
which are injected into Jinja templates via the format_for_jinja() function.
"""

CANONICAL_TAGS = {
    'opening_moves': {
        'opens_with_concession': 'grants a point before pivoting',
        'opens_with_scene': 'begins with concrete situation or image',
        'opens_with_question': 'leads with a question to frame what follows',
        'opens_with_declaration': 'starts with bold, direct claim',
        'opens_with_paradox': 'begins with apparent contradiction',
    },
    'building_moves': {
        'builds_via_examples': 'accumulates concrete instances',
        'builds_via_contrast': 'develops by showing what it\'s not',
        'builds_via_analogy': 'explains through comparison',
        'builds_via_escalation': 'each sentence raises stakes',
        'builds_via_refinement': 'progressively sharpens a claim',
        'layers_parallel_structure': 'repeated syntax for emphasis',
    },
    'pivot_moves': {
        'pivots_with_but': 'concedes then reverses',
        'pivots_via_question': 'question shifts direction',
        'pivots_from_general_to_specific': 'abstract to concrete',
        'pivots_from_specific_to_general': 'concrete to principle',
        'returns_to_opening': 'circles back to earlier theme',
    },
    'closing_moves': {
        'closes_with_implication': 'ends by gesturing to consequences',
        'closes_with_short_sentence': 'punchy conclusion',
        'closes_with_question': 'leaves reader pondering',
        'closes_with_restatement': 'echoes opening with new weight',
        'closes_by_widening_scope': 'pulls back to broader significance',
    },
    'texture_moves': {
        'varies_sentence_length': 'alternates long and short',
        'embeds_qualification': 'weaves caveats smoothly',
        'pairs_abstract_and_concrete': 'grounds ideas tangibly',
        'uses_periodic_structure': 'delays main clause',
        'uses_cumulative_structure': 'main clause first, then elaborates',
        'balances_clauses': 'symmetrical phrasing',
        'deploys_list_with_variation': 'lists that avoid monotony',
    },
    'stance_moves': {
        'acknowledges_complexity': 'admits difficulty without retreating',
        'asserts_with_confidence': 'states position without over-hedging',
        'invites_reader_in': 'creates shared inquiry',
        'addresses_skeptic': 'anticipates and answers doubts',
    },
}


def get_all_canonical_tags() -> set:
    """Return set of all canonical tag names for filtering.

    This function is used to filter out canonical tags when retrieving
    author-specific (Tier 2) tags from the segment catalog.

    Returns:
        Set of canonical tag names (e.g., {'opens_with_concession', ...})

    Example:
        >>> canonical_set = get_all_canonical_tags()
        >>> all_tags = ['opens_with_concession', 'author_specific_tag']
        >>> tier2_tags = [t for t in all_tags if t not in canonical_set]
        >>> print(tier2_tags)
        ['author_specific_tag']
    """
    tags = set()
    for category_tags in CANONICAL_TAGS.values():
        tags.update(category_tags.keys())
    return tags


def get_canonical_tags_by_category() -> dict:
    """Return canonical tags organized by category (for display).

    Returns:
        Dictionary mapping category keys to tag dicts.
        Example: {'opening_moves': {'opens_with_concession': '...', ...}, ...}
    """
    return CANONICAL_TAGS.copy()


def format_for_jinja() -> list[dict]:
    """Format canonical tags for Jinja template injection.

    Transforms the nested CANONICAL_TAGS dictionary into a format
    suitable for Jinja2 template iteration.

    Returns:
        List of dicts with:
        - category: category name in Title Case (e.g., "Opening Moves")
        - tags: list of (tag_name, description) tuples

    Example:
        >>> formatted = format_for_jinja()
        >>> formatted[0]
        {
            'category': 'Opening Moves',
            'tags': [
                ('opens_with_concession', 'grants a point before pivoting'),
                ('opens_with_scene', 'begins with concrete situation or image'),
                ...
            ]
        }

    Usage in template:
        {% for category_data in canonical_tags_formatted %}
        **{{ category_data.category }}:**
        {% for tag, description in category_data.tags %}
        - `{{ tag }}` — {{ description }}
        {% endfor %}
        {% endfor %}
    """
    formatted = []
    for category_key, tags in CANONICAL_TAGS.items():
        # Convert snake_case to Title Case
        category_name = category_key.replace('_', ' ').title()
        tag_list = [(tag, desc) for tag, desc in tags.items()]
        formatted.append({
            'category': category_name,
            'tags': tag_list
        })
    return formatted
