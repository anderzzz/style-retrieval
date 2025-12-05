from abc import ABC, abstractmethod
from pydantic import BaseModel


class ToolConfig(BaseModel):
    """Base configuration for all tools."""
    name: str
    description: str


class Tool(ABC):
    """Abstract base for LLM-callable tools."""

    def __init__(self, config: ToolConfig):
        self.config = config

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass

    @abstractmethod
    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling schema."""
        pass
