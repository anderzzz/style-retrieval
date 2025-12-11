"""
Simple LLM wrapper for text completion.

Provides a clean interface for text-in, text-out interactions with any
LiteLLM-supported model.
"""
from typing import Optional, Union, Type, Any, Literal
import json
from enum import Enum
from pydantic import BaseModel, ValidationError, Field
import litellm

# Enable client-side JSON schema validation for providers that don't natively support it
litellm.enable_json_schema_validation = True

from belletrist.tools import Tool


class LLMRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """Represents a single message in a conversation."""
    role: LLMRole
    content: str
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to LiteLLM-compatible dictionary."""
        msg = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            msg["name"] = self.name
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


class LLMResponse(BaseModel):
    """Standardized response from any LLM call."""
    model_config = {"arbitrary_types_allowed": True}

    content: str | BaseModel | None = None
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None
    model: str | None = None
    usage: dict | None = None
    raw_response: Any = None
    schema_validation_mode: Literal["strict", "fallback", "none"] | None = None
    validation_attempted: bool = False

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


class LLMConfig(BaseModel):
    """Configuration for LLM invocation."""
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    timeout: int | None = Field(default=None, gt=0)
    api_key: str
    api_base: str | None = Field(default=None)
    response_format: dict | None = Field(
        default=None,
        description="Structured output format. Use {'type': 'json_object'} for JSON mode."
    )
    extra_params: dict = Field(default_factory=dict)


class LLM:
    """
    Simple LLM wrapper for text completion.

    Handles basic text-in, text-out interactions with any LiteLLM-supported model.
    Focuses on simplicity and clarity over feature completeness.
    """

    def __init__(self, config: Union[str, LLMConfig], **kwargs):
        """
        Initialize the LLM with configuration.

        Args:
            config: Either an LLMConfig object or a model string
            **kwargs: If config is a string, these become config parameters

        Example:
            # Explicit config (recommended, matches PromptMaker pattern)
            llm = LLM(LLMConfig(model="gpt-4", temperature=0.7))

            # Shorthand for notebooks
            llm = LLM("gpt-4", temperature=0.7)
        """
        if isinstance(config, str):
            # Shorthand: model string + kwargs
            self.config = LLMConfig(model=config, **kwargs)
        else:
            # Explicit: LLMConfig object (preferred)
            self.config = config

    def complete(
            self,
            prompt: Union[str, list[Message]],
            system: Optional[str] = None,
            **kwargs
    ) -> LLMResponse:
        """
        Execute a single completion.

        Args:
            prompt: Either a string (converted to user message) or list of Messages
            system: Optional system prompt to prepend
            **kwargs: Override config parameters for this call

        Returns:
            LLMResponse containing the model's output

        Example:
            llm = LLM("gpt-4")
            response = llm.complete("What is the capital of France?")
            print(response.content)
        """
        # Convert string prompt to message list
        if isinstance(prompt, str):
            messages = [Message(role=LLMRole.USER, content=prompt)]
        else:
            messages = list(prompt)

        # Prepend system message if provided
        if system:
            messages.insert(0, Message(role=LLMRole.SYSTEM, content=system))

        # Build the request
        request_params = self._build_request_params(messages, **kwargs)

        # Execute the completion
        raw_response = litellm.completion(**request_params)

        # Parse and return the response
        return self._parse_response(raw_response)

    def complete_json(
            self,
            prompt: Union[str, list[Message]],
            system: Optional[str] = None,
            **kwargs
    ) -> LLMResponse:
        """
        Execute a completion with JSON mode enabled.

        Convenience method that sets response_format={'type': 'json_object'}
        to request structured JSON output from the model.

        Args:
            prompt: Either a string (converted to user message) or list of Messages
            system: Optional system prompt to prepend
            **kwargs: Override config parameters for this call

        Returns:
            LLMResponse containing JSON content

        Example:
            llm = LLM("gpt-4")
            response = llm.complete_json("Return user info as JSON: name=John, age=30")
            data = json.loads(response.content)
        """
        return self.complete(prompt, system, response_format={"type": "json_object"}, **kwargs)

    def complete_with_schema(
            self,
            prompt: Union[str, list[Message]],
            schema_model: Type[BaseModel],
            system: Optional[str] = None,
            strict: bool = True,
            **kwargs
    ) -> LLMResponse:
        """
        Execute a completion with schema-validated structured output.

        Uses LiteLLM's JSON schema mode to enforce structure at the LLM level.
        Falls back to json_object mode with manual validation if strict schemas
        aren't supported by the provider.

        Args:
            prompt: Either a string (converted to user message) or list of Messages
            schema_model: Pydantic model class defining the expected JSON structure
            system: Optional system prompt to prepend
            strict: Whether to use strict schema mode (default: True). If False,
                   uses json_object mode with manual validation
            **kwargs: Override config parameters for this call

        Returns:
            LLMResponse with validated Pydantic model instance in .content field

        Raises:
            ValueError: If LLM returns invalid JSON or response fails schema validation

        Example:
            from pydantic import BaseModel, Field

            class UserInfo(BaseModel):
                name: str
                age: int = Field(..., ge=0, le=120)

            llm = LLM("gpt-4")
            response = llm.complete_with_schema(
                "Return info for user John, age 30",
                UserInfo
            )
            user = response.content  # Already a validated UserInfo instance
            print(user.name, user.age)
        """
        # Extract JSON schema from Pydantic model
        json_schema = schema_model.model_json_schema()

        # Build response_format for LiteLLM
        if strict:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "schema": json_schema,
                    "strict": True
                }
            }
        else:
            response_format = {"type": "json_object"}

        # Attempt LLM call with schema
        validation_mode = "strict" if strict else "fallback"
        try:
            response = self.complete(prompt, system, response_format=response_format, **kwargs)
        except Exception as e:
            # If strict schema fails (provider doesn't support it OR validation fails), retry with json_object mode
            error_msg = str(e).lower()
            # Check for schema-related errors OR jsonschema ValidationError
            is_schema_error = ("schema" in error_msg or "strict" in error_msg or
                             "json_schema" in error_msg or "validationerror" in error_msg or
                             "is a required property" in error_msg)

            if strict and is_schema_error:
                # Fallback to json_object mode (less strict, let Pydantic validate instead)
                response = self.complete(prompt, system, response_format={"type": "json_object"}, **kwargs)
                validation_mode = "fallback"
            else:
                # Different error, re-raise
                raise

        # Extract JSON from response (handle markdown wrapping)
        content = response.content.strip()

        # Check if JSON is wrapped in markdown code blocks
        if content.startswith("```"):
            # Extract content between ``` markers
            lines = content.split('\n')
            # Skip first line (```json or ```)
            start_idx = 1
            # Find closing ```
            end_idx = len(lines)
            for i in range(1, len(lines)):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            content = '\n'.join(lines[start_idx:end_idx])

        # Parse and validate response against schema
        try:
            parsed_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON for {schema_model.__name__} schema.\n"
                f"Error: {e}\n"
                f"Response preview: {response.content[:500]}\n"
                f"This might indicate the response was truncated due to token limits.\n"
                f"Consider increasing max_tokens in LLMConfig or reducing the amount of data requested."
            ) from e

        # Unwrap common wrapper patterns (some LLMs add these in json_object mode)
        # Check if response is wrapped in a single-key object
        if isinstance(parsed_data, dict) and len(parsed_data) == 1:
            wrapper_key = list(parsed_data.keys())[0]
            # Common wrapper keys: json_payload, data, response, result, output, or schema name
            common_wrappers = {'json_payload', 'data', 'response', 'result', 'output',
                             schema_model.__name__, schema_model.__name__.lower()}
            if wrapper_key in common_wrappers:
                parsed_data = parsed_data[wrapper_key]

        try:
            validated_model = schema_model(**parsed_data)
        except ValidationError as e:
            # Extract helpful details about what's missing
            error_details = []
            for error in e.errors():
                field_path = " -> ".join(str(x) for x in error['loc'])
                error_details.append(f"  - {field_path}: {error['msg']} (got: {error.get('input', 'missing')})")

            raise ValueError(
                f"LLM response failed schema validation for {schema_model.__name__}.\n"
                f"Validation errors:\n" + "\n".join(error_details) + "\n\n"
                f"This usually means the LLM:\n"
                f"  1. Omitted required fields (common with weaker models)\n"
                f"  2. Used wrong data types\n"
                f"  3. Violated field constraints\n\n"
                f"Consider:\n"
                f"  - Using a model with better structured output support (GPT-4, Claude, Gemini)\n"
                f"  - Increasing temperature if the model is being too conservative\n"
                f"  - Simplifying the schema if it's too complex\n\n"
                f"Response preview: {response.content[:500]}"
            ) from e

        # Update response with validated model and metadata
        response.content = validated_model
        response.schema_validation_mode = validation_mode
        response.validation_attempted = True

        return response

    def _build_request_params(self, messages: list[Message], **overrides) -> dict:
        """
        Build parameters for the LiteLLM completion call.

        Args:
            messages: List of messages to send
            **overrides: Parameters to override from config

        Returns:
            Dictionary of parameters for litellm.completion()
        """
        # Start with config defaults
        params = {
            "model": self.config.model,
            "messages": [msg.model_dump(exclude_none=True) for msg in messages],
            "temperature": self.config.temperature,
        }

        # Add optional config parameters if set
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if self.config.top_p != 1.0:
            params["top_p"] = self.config.top_p
        if self.config.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty != 0.0:
            params["presence_penalty"] = self.config.presence_penalty
        if self.config.timeout:
            params["timeout"] = self.config.timeout
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.api_base:
            params["api_base"] = self.config.api_base
        if self.config.response_format:
            params["response_format"] = self.config.response_format

        # Apply any overrides for this specific call
        params.update(overrides)

        return params

    def _parse_response(self, raw_response) -> LLMResponse:
        """
        Parse raw LiteLLM response into our standardized format.

        Args:
            raw_response: Raw response from litellm.completion()

        Returns:
            Standardized LLMResponse object
        """
        choice = raw_response.choices[0]
        message = choice.message

        # Extract usage information if available
        usage = None
        if hasattr(raw_response, 'usage'):
            usage = {
                "prompt_tokens": raw_response.usage.prompt_tokens,
                "completion_tokens": raw_response.usage.completion_tokens,
                "total_tokens": raw_response.usage.total_tokens
            }

        return LLMResponse(
            content=getattr(message, 'content', None),
            tool_calls=getattr(message, 'tool_calls', None),
            finish_reason=choice.finish_reason,
            model=raw_response.model,
            usage=usage,
            raw_response=raw_response
        )


class ToolLLM(LLM):
    """
    LLM with tool execution capabilities.

    Manages tool registration, execution, and the control flow between
    LLM responses and tool calls.
    """

    def __init__(self, config: Union[str, LLMConfig], **kwargs):
        """
        Initialize the tool-enabled LLM.

        Args:
            config: Either an LLMConfig object or a model string
            **kwargs: If config is a string, these become config parameters

        Example:
            # Explicit config
            llm = ToolLLM(LLMConfig(model="gpt-4", temperature=0.7))

            # Shorthand
            llm = ToolLLM("gpt-4", temperature=0.7)
        """
        super().__init__(config, **kwargs)
        self.tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """
        Register a tool for use by the LLM.

        Args:
            tool: Tool instance to register

        Example:
            llm = ToolLLM("gpt-4")
            llm.register_tool(WordCountTool())
        """
        self.tools[tool.config.name] = tool

    def complete_with_tools(
            self,
            prompt: str,
            system: Optional[str] = None,
            max_iterations: int = 5,
            **kwargs
    ) -> str:
        """
        Execute completion with automatic tool handling.

        The LLM will automatically invoke registered tools as needed to
        answer the prompt, handling multiple rounds of tool calls if necessary.

        Args:
            prompt: User's question or request
            system: Optional system prompt
            max_iterations: Maximum rounds of tool execution
            **kwargs: Override config parameters

        Returns:
            Final text response after all tool executions

        """
        # Initialize conversation with user prompt
        messages = []
        if system:
            messages.append(Message(role=LLMRole.SYSTEM, content=system))
        messages.append(Message(role=LLMRole.USER, content=prompt))

        # Get tool schemas for the LLM
        tool_schemas = [tool.to_openai_schema() for tool in self.tools.values()]

        # Tool execution loop
        for iteration in range(max_iterations):
            # Get LLM response with tool schemas
            response = self._call_llm_with_tools(messages, tool_schemas, **kwargs)

            # If no tool calls, we have our final answer
            if not response.tool_calls:
                return response.content or "No response generated."

            # Add assistant's message (with tool calls) to history
            messages.append(
                Message(
                    role=LLMRole.ASSISTANT,
                    content=response.content or "",
                    tool_calls=response.tool_calls
                )
            )

            # Execute each tool call and add results
            for tool_call in response.tool_calls:
                tool_result = self._execute_tool_call(tool_call)
                messages.append(
                    Message(
                        role=LLMRole.TOOL,
                        content=tool_result['output'],
                        tool_call_id=tool_call['id'],
                        name=tool_result['name']
                    )
                )

        # If we've exhausted iterations, return last response
        return response.content or "Maximum iterations reached without final answer."

    def _call_llm_with_tools(
            self,
            messages: list[Message],
            tool_schemas: list[dict],
            **kwargs
    ) -> LLMResponse:
        """
        Call the LLM with tool schemas enabled.

        Args:
            messages: Conversation history
            tool_schemas: OpenAI-format tool definitions
            **kwargs: Override config parameters

        Returns:
            LLMResponse potentially containing tool calls
        """
        # Build request parameters
        params = self._build_request_params(messages, **kwargs)

        # Add tools if available
        if tool_schemas:
            params['tools'] = tool_schemas
            params['tool_choice'] = 'auto'  # Let the model decide when to use tools

        # Execute the completion
        raw_response = litellm.completion(**params)

        # Parse and return
        return self._parse_response(raw_response)

    def _execute_tool_call(self, tool_call: dict) -> dict:
        """
        Execute a single tool call.

        Args:
            tool_call: Tool call from LLM response

        Returns:
            Dictionary with tool execution results
        """
        function_name = tool_call['function']['name']
        arguments_str = tool_call['function']['arguments']

        # Parse arguments (they come as JSON string)
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            return {
                'name': function_name,
                'output': f"Error: Failed to parse arguments for {function_name}"
            }

        # Execute the tool
        if function_name in self.tools:
            try:
                tool = self.tools[function_name]
                output = tool.execute(**arguments)
            except Exception as e:
                output = f"Error executing {function_name}: {str(e)}"
        else:
            output = f"Error: Tool '{function_name}' not found"

        return {
            'name': function_name,
            'output': str(output)
        }

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a registered tool by name."""
        return self.tools.get(name)
