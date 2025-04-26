{{ ... }}

## Using `RunnableLambda` for Custom Logic

When building LangChain Expression Language (LCEL) pipelines, we often need to perform custom data manipulation, filtering, or other operations that don't map directly to standard LangChain components (like Prompts, LLMs, or Parsers).

To integrate arbitrary Python functions seamlessly into the LCEL pipe (`|`) syntax, we use `langchain_core.runnables.RunnableLambda`. This allows us to wrap a standard Python function, making it a valid step within the chain.

**Example Use Cases:**

*   Filtering data based on custom criteria (e.g., `filter_short_translated_utterance`).
*   Loading data from specific sources or formats within the chain.
*   Performing complex data transformations between steps.
*   Injecting constant values or context needed by subsequent steps (often used with `RunnablePassthrough.assign`).

By using `RunnableLambda`, we maintain the declarative and readable structure of LCEL chains while incorporating necessary custom Python logic.