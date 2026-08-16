# Repository conventions

## Tool declarations

- Declare every LangChain tool with `@tool` from `langchain_core.tools`.
- Do not construct tools with `StructuredTool.from_function` or equivalent manual wrappers.
- When a tool needs runtime dependencies, define the decorated function inside a factory closure and return the decorated tool.
- Give every tool function complete type annotations and a docstring that explains when to use it and what its arguments mean.
