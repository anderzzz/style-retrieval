"""
Prompt maker that constructs prompt snippets from Jinja templates.

Uses Pydantic models for type-safe, validated prompt construction.

"""
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from belletrist.prompts import BasePromptConfig


PROMPTS_PATH = (Path(__file__).parent / "prompts" / "templates").resolve()

class PromptMaker:
    """Constructs prompt snippets from Jinja templates using Pydantic models."""

    def __init__(self):
        """Initialize the prompt maker with Jinja environment."""
        self.env = Environment(
            loader=FileSystemLoader(PROMPTS_PATH),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def render(self, prompt_model: BasePromptConfig) -> str:
        """
        Render a prompt snippet from a Pydantic model.

        Args:
            prompt_model: Pydantic model containing validated template variables

        Returns:
            Rendered prompt snippet as a string

        Raises:
            pydantic.ValidationError: If model has invalid/missing fields
            jinja2.TemplateNotFound: If template file doesn't exist

        Example:
            maker = PromptMaker()
            config = PreambleTextConfig(text_to_analyze="Sample text")
            prompt = maker.render(config)
        """
        template_name = prompt_model.template_name() + ".jinja"
        template_vars = prompt_model.model_dump()

        template = self.env.get_template(template_name)
        return template.render(**template_vars)
