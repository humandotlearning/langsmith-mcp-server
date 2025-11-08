"""Registration module for LangSmith MCP tools."""

import json
from typing import Any, Dict, List, Optional, Union

from fastmcp import FastMCP
from fastmcp.server import Context

from langsmith_mcp_server.common.helpers import get_client_from_context
from langsmith_mcp_server.services.tools.datasets import (
    list_datasets_tool,
    list_examples_tool,
    read_dataset_tool,
    read_example_tool,
)
from langsmith_mcp_server.services.tools.prompts import (
    get_prompt_tool,
    list_prompts_tool,
)
from langsmith_mcp_server.services.tools.traces import (
    fetch_runs_tool,
    fetch_trace_tool,
    get_project_runs_stats_tool,
    get_thread_history_tool,
    list_projects_tool,
)


def register_tools(mcp: FastMCP) -> None:
    """
    Register all LangSmith tool-related functionality with the MCP server.
    This function configures and registers various tools for interacting with LangSmith,
    including prompt management, conversation history, traces, and analytics.

    Args:
        mcp: The MCP server instance to register tools with
    """

    @mcp.tool()
    def list_prompts(is_public: str = "false", limit: int = 20, ctx: Context = None) -> Dict[str, Any]:
        """
        Fetch prompts from LangSmith with optional filtering.

        Args:
            is_public (str): Filter by prompt visibility - "true" for public prompts,
                            "false" for private prompts (default: "false")
            limit (int): Maximum number of prompts to return (default: 20)

        Returns:
            Dict[str, Any]: Dictionary containing the prompts and metadata
        """
        try:
            client = get_client_from_context(ctx)
            is_public_bool = is_public.lower() == "true"
            return list_prompts_tool(client, is_public_bool, limit)
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def get_prompt_by_name(prompt_name: str, ctx: Context = None) -> Dict[str, Any]:
        """
        Get a specific prompt by its exact name.

        Args:
            prompt_name (str): The exact name of the prompt to retrieve
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the prompt details and template,
                          or an error message if the prompt cannot be found
        """
        try:
            client = get_client_from_context(ctx)
            return get_prompt_tool(client, prompt_name=prompt_name)
        except Exception as e:
            return {"error": str(e)}

    # Register conversation tools
    @mcp.tool()
    def get_thread_history(thread_id: str, project_name: str, ctx: Context = None) -> Dict[str, Any]:
        """
        Retrieve the message history for a specific conversation thread.

        Args:
            thread_id (str): The unique ID of the thread to fetch history for
            project_name (str): The name of the project containing the thread
                               (format: "owner/project" or just "project")

        Returns:
            Dict[str, Any]: Dictionary containing the thread history,
                                or an error message if the thread cannot be found
        """
        try:
            client = get_client_from_context(ctx)
            return get_thread_history_tool(client, thread_id, project_name)
        except Exception as e:
            return {"error": str(e)}

    # Register analytics tools
    @mcp.tool()
    def get_project_runs_stats(project_name: str = None, trace_id: str = None, ctx: Context = None) -> Dict[str, Any]:
        """
        Get statistics about runs in a LangSmith project.

        Args:
            project_name (str): The name of the project to analyze
                              (format: "owner/project" or just "project")
            trace_id (str): The specific ID of the trace to fetch (preferred parameter)

        Returns:
            Dict[str, Any]: Dictionary containing the requested project run statistics
                          or an error message if statistics cannot be retrieved
        """
        try:
            client = get_client_from_context(ctx)
            return get_project_runs_stats_tool(client, project_name, trace_id)
        except Exception as e:
            return {"error": str(e)}

    # # Register trace tools
    # @mcp.tool()
    # def fetch_trace(project_name: str = None, trace_id: str = None, ctx: Context = None) -> Dict[str, Any]:
    #     """
    #     Fetch trace content for debugging and analyzing LangSmith runs.

    #     Note: Only one parameter (project_name or trace_id) is required.
    #     If both are provided, trace_id is preferred.
    #     String "null" inputs are handled as None values.

    #     Args:
    #         project_name (str, optional): The name of the project to fetch the latest trace from
    #         trace_id (str, optional): The specific ID of the trace to fetch (preferred parameter)

    #     Returns:
    #         Dict[str, Any]: Dictionary containing the trace data and metadata,
    #                       or an error message if the trace cannot be found
    #     """
    #     try:
    #         client = get_client_from_context(ctx)
    #         return fetch_trace_tool(client, project_name, trace_id)
    #     except Exception as e:
    #         return {"error": str(e)}

    @mcp.tool()
    def fetch_runs(
        project_name: str,
        trace_id: Optional[str] = None,
        run_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        reference_example_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        error: Optional[str] = None,
        run_ids: Optional[str] = None,
        is_root: Optional[str] = None,
        filter: Optional[str] = None,
        trace_filter: Optional[str] = None,
        tree_filter: Optional[str] = None,
        order_by: str = "-start_time",
        limit: str = "50",
        select: Optional[str] = None,
        show_trace_tree: str = "true",
        trace_tree_depth: str = "0",
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch LangSmith runs (traces, tools, chains, etc.) from one or more projects
        using flexible filters, query language expressions, and trace-level constraints.

        ---
        🧩 PURPOSE
        ----------
        This is a **general-purpose LangSmith run fetcher** designed for analytics,
        trace export, and automated exploration.

        It wraps `client.list_runs()` with complete support for:
        - Multiple project names or IDs
        - The **Filter Query Language (FQL)** for precise queries
        - Hierarchical filtering across trace trees
        - Sorting, field selection, and ID-based retrieval

        It returns **raw `dict` objects** suitable for further analysis or export.

        ---
        ⚙️ PARAMETERS
        -------------
        project_name : str
            The project name to fetch runs from. For multiple projects, use JSON array string (e.g., '["project1", "project2"]').

        trace_id : str, optional
            Return only runs that belong to a specific trace tree.

        run_type : str, optional
            Filter runs by type (e.g. "llm", "chain", "tool", "retriever").

        dataset_name : str, optional
            Return runs associated with a specific dataset name.

        reference_example_id : str, optional
            Return runs linked to a specific dataset example (used for model comparison).

        parent_run_id : str, optional
            Return runs that are **children** of the specified run.
            Useful for fetching agent trajectories or grouped subruns.

        error : str, optional
            Filter by error status: "true" for errored runs, "false" for successful runs.

        run_ids : str, optional
            Directly fetch runs by their IDs. Can be a single ID or JSON array like '["id1", "id2"]'.
            ⚠️ If this is provided, **all other filters are ignored.**

        is_root : str, optional
            Filter root traces: "true" for only top-level traces, "false" to exclude roots.
            If not provided, returns all runs.

        filter : str, optional
            A **Filter Query Language (FQL)** expression that filters runs by fields,
            metadata, tags, feedback, latency, or time.

            ─── Common field names ───
            - `id`, `name`, `run_type`
            - `start_time`, `end_time`
            - `latency`
            - `total_tokens`
            - `error`
            - `tags`
            - `feedback_key`, `feedback_score`
            - `metadata_key`, `metadata_value`
            - `execution_order`

            ─── Supported comparators ───
            - `eq`, `neq` → equal / not equal
            - `gt`, `gte`, `lt`, `lte` → numeric or time comparisons
            - `has` → tag or metadata contains value
            - `search` → substring or full-text match
            - `and`, `or`, `not` → logical operators

            ─── Examples ───
            ```python
            'gt(latency, "5s")'                                # took longer than 5 seconds
            'neq(error, null)'                                  # errored runs
            'has(tags, "beta")'                                 # runs tagged "beta"
            'and(eq(name,"ChatOpenAI"), eq(run_type,"llm"))'    # named & typed runs
            'search("image classification")'                    # full-text search
            ```

        trace_filter : str, optional
            Filter applied **to the root run** in each trace tree.
            Lets you select child runs based on root attributes or feedback.

            Example:
            ```python
            'and(eq(feedback_key,"user_score"), eq(feedback_score,1))'
            ```
            → return runs whose root trace has a user_score of 1.

        tree_filter : str, optional
            Filter applied **to any run** in the trace tree (including siblings or children).
            Example:
            ```python
            'eq(name,"ExpandQuery")'
            ```
            → return runs if *any* run in their trace had that name.

        order_by : str, default "-start_time"
            Sort field; prefix with "-" for descending order.

        limit : str, default "50"
            Maximum number of runs to return (as string, e.g., "50").

        select : str, optional
            Fields to return as JSON array string (e.g., '["id", "name", "inputs"]').
            By default, all fields are returned.
            
            This function performs custom field filtering after fetching the data,
            giving you full flexibility to select any top-level fields. Fields are
            filtered after data conversion, so you get the complete nested structure
            for selected fields (which can then be summarized if `show_trace_tree=True`).
            
            Available outer field names:
            - "id" - Unique identifier for the span (UUID)
            - "name" - The name associated with the run
            - "inputs" - A map or set of inputs provided to the run
            - "run_type" - Type of run, e.g., "llm", "chain", "tool"
            - "start_time" - Start time of the run (datetime)
            - "end_time" - End time of the run (datetime)
            - "extra" - Any extra information run
            - "error" - Error message if the run encountered an error
            - "outputs" - A map or set of outputs generated by the run
            - "events" - A list of event objects associated with the run (for streaming)
            - "tags" - Tags or labels associated with the run (array of strings)
            - "trace_id" - Unique identifier for the trace (UUID)
            - "dotted_order" - Ordering string, hierarchical
            - "status" - Current status of the run execution, e.g., "error", "pending", "success"
            - "child_run_ids" - List of IDs for all child runs (array of UUIDs)
            - "direct_child_run_ids" - List of IDs for direct children of this run (array of UUIDs)
            - "parent_run_ids" - List of IDs for all parent runs (array of UUIDs)
            - "parent_run_id" - ID of the parent run (UUID)
            - "feedback_stats" - Aggregations of feedback statistics for this run
            - "reference_example_id" - ID of a reference example associated with the run (UUID)
            - "total_tokens" - Total number of tokens processed by the run (integer)
            - "prompt_tokens" - Number of tokens in the prompt of the run (integer)
            - "completion_tokens" - Number of tokens in the completion of the run (integer)
            - "total_cost" - Total cost associated with processing the run (string)
            - "prompt_cost" - Cost associated with the prompt part of the run (string)
            - "completion_cost" - Cost associated with the completion of the run (string)
            - "first_token_time" - Time when the first token was generated (datetime, for streaming LLM runs)
            - "session_id" - Session identifier for the run, also known as the tracing project ID
            - "in_dataset" - Indicates whether the run is included in a dataset (boolean)
            
            Example:
            ```python
            fetch_runs("my-project", select=["id", "name", "inputs", "outputs", "total_tokens"])
            ```
            
            Note: When `show_trace_tree=True`, selected fields will still show the tree structure
            with summary metrics for nested content.

        show_trace_tree : str, default "true"
            If "true", returns a simplified trace tree structure showing top-level fields
            with metrics for nested content (field count, character count) instead of
            full nested data. Useful for exploring large traces without loading all content.
            If "false", returns full run data as before.
            
            When enabled, nested dictionaries and lists are replaced with summary objects
            containing `_type`, `_field_count`, `_character_count`, and preview information.
            Use `trace_tree_depth` to control how many levels of nested content are shown
            before summarizing.

        trace_tree_depth : str, default "0"
            Controls how many levels deep to show actual content before summarizing when
            `show_trace_tree="true"`. This allows you to explore nested structures at
            different levels of detail (as string, e.g., "0", "1", "2").
            
            - "0" (default): Summarize all nested structures immediately (shows only metrics)
            - "1": Show one level of nested content, then summarize deeper levels
            - "2": Show two levels of nested content, then summarize deeper levels
            - etc.
            
            Example:
            ```python
            # Show outputs with 2 levels of detail
            fetch_runs("my-project", select='["outputs"]', 
                    show_trace_tree="true", trace_tree_depth="2")
            ```

        ---
        📤 RETURNS
        ----------
        List[dict]
            A list of LangSmith `dict` objects that satisfy the query.

        ---
        🧪 EXAMPLES
        ------------
        1️⃣ **Get latest 10 root runs**
        ```python
        runs = fetch_runs("alpha-project", is_root="true", limit="10")
        ```

        2️⃣ **Get all tool runs that errored**
        ```python
        runs = fetch_runs("alpha-project", run_type="tool", error="true")
        ```

        3️⃣ **Get all runs that took >5s and have tag "experimental"**
        ```python
        runs = fetch_runs("alpha-project", filter='and(gt(latency,"5s"), has(tags,"experimental"))')
        ```

        4️⃣ **Get all runs in a specific conversation thread**
        ```python
        thread_id = "abc-123"
        fql = f'and(in(metadata_key, ["session_id","conversation_id","thread_id"]), eq(metadata_value, "{thread_id}"))'
        runs = fetch_runs("alpha-project", is_root="true", filter=fql)
        ```

        5️⃣ **List all runs called "extractor" whose root trace has feedback user_score=1**
        ```python
        runs = fetch_runs(
            "alpha-project",
            filter='eq(name,"extractor")',
            trace_filter='and(eq(feedback_key,"user_score"), eq(feedback_score,1))'
        )
        ```

        6️⃣ **List all runs that started after a timestamp and either errored or got low feedback**
        ```python
        fql = 'and(gt(start_time,"2023-07-15T12:34:56Z"), or(neq(error,null), and(eq(feedback_key,"Correctness"), eq(feedback_score,0.0))))'
        runs = fetch_runs("alpha-project", filter=fql)
        ```

        7️⃣ **Get specific fields with trace tree summary**
        ```python
        runs = fetch_runs(
            "alpha-project",
            select='["id", "name", "inputs", "outputs", "total_tokens", "total_cost"]',
            show_trace_tree="true",
            trace_tree_depth="1",  # Show 1 level deep, then summarize
            limit="10"
        )
        ```
        → Returns only selected fields with nested content shown as summary metrics.
        
        8️⃣ **Explore nested outputs with custom depth**
        ```python
        runs = fetch_runs(
            "alpha-project",
            select='["outputs"]',
            show_trace_tree="true",
            trace_tree_depth="2",  # Show 2 levels deep to see nested structure
            limit="5"
        )
        ```
        → Shows outputs with 2 levels of nested content visible before summarizing.

        ---
        🧠 NOTES FOR AGENTS
        --------------------
        - Use this to **query LangSmith data sources dynamically**.
        - Compose FQL strings programmatically based on your intent.
        - Combine `filter`, `trace_filter`, and `tree_filter` for hierarchical logic.
        - Try to ALWASY use filters and trace_filter and tree_filter first before using show_trace_tree and trace_tree_depth.
        - Always verify that `project_name` matches an existing LangSmith project.
        - Returned `dict` objects have fields like:
        - `id`, `name`, `run_type`, `inputs`, `outputs`, `error`, `start_time`, `end_time`, `latency`, `metadata`, `feedback`, etc.
        """
        try:
            client = get_client_from_context(ctx)
            
            # Parse project_name - can be a single string or JSON array
            parsed_project_name = project_name
            if project_name and project_name.startswith("["):
                try:
                    parsed_project_name = json.loads(project_name)
                except json.JSONDecodeError:
                    pass  # Use as-is if not valid JSON
            
            # Parse boolean strings
            parsed_error = None
            if error is not None:
                parsed_error = error.lower() == "true" if error.lower() in ("true", "false") else None
            
            parsed_is_root = None
            if is_root is not None:
                if is_root.lower() == "true":
                    parsed_is_root = True
                elif is_root.lower() == "false":
                    parsed_is_root = False
            
            parsed_show_trace_tree = show_trace_tree.lower() == "true"
            
            # Parse list strings (JSON arrays)
            parsed_run_ids = None
            if run_ids is not None:
                try:
                    parsed_run_ids = json.loads(run_ids) if run_ids.startswith("[") else [run_ids]
                except (json.JSONDecodeError, AttributeError):
                    parsed_run_ids = [run_ids] if run_ids else None
            
            parsed_select = None
            if select is not None:
                try:
                    parsed_select = json.loads(select) if select.startswith("[") else [select]
                except (json.JSONDecodeError, AttributeError):
                    parsed_select = [select] if select else None
            
            # Parse integer strings
            parsed_limit = int(limit) if limit else 50
            parsed_trace_tree_depth = int(trace_tree_depth) if trace_tree_depth else 0
            
            return fetch_runs_tool(
                client,
                project_name=parsed_project_name,
                trace_id=trace_id,
                run_type=run_type,
                dataset_name=dataset_name,
                reference_example_id=reference_example_id,
                parent_run_id=parent_run_id,
                error=parsed_error,
                run_ids=parsed_run_ids,
                is_root=parsed_is_root,
                filter=filter,
                trace_filter=trace_filter,
                tree_filter=tree_filter,
                order_by=order_by,
                limit=parsed_limit,
                select=parsed_select,
                show_trace_tree=False,
                trace_tree_depth=5,
            )
        except Exception as e:
            return {"error": str(e)}

    # Register project tools
    @mcp.tool()
    def list_projects(limit: str = "5", project_name: str = None, more_info: str = "false", ctx: Context = None) -> Dict[str, Any]:
        """
        List LangSmith projects with optional filtering and detail level control.
        
        Fetches projects from LangSmith, optionally filtering by name and controlling
        the level of detail returned. Can return either simplified project information
        or full project details.
        
        ---
        🧩 PURPOSE
        ----------
        This function provides a convenient way to list and explore LangSmith projects.
        It supports:
        - Filtering projects by name (partial match)
        - Limiting the number of results
        - Choosing between simplified or full project information
        - Automatically extracting deployment IDs from nested project data
        
        ---
        ⚙️ PARAMETERS
        -------------
        limit : str, default "5"
            Maximum number of projects to return (as string, e.g., "5"). This can be adjusted by agents
            or users based on their needs.
        
        project_name : str, optional
            Filter projects by name using partial matching. If provided, only projects
            whose names contain this string will be returned.
            Example: `project_name="Chat"` will match "Chat-LangChain", "ChatBot", etc.
        
        more_info : str, default "false"
            Controls the level of detail returned:
            - `"false"` (default): Returns simplified project information with only
            essential fields: `name`, `project_id`, and `agent_deployment_id` (if available)
            - `"true"`: Returns full project details as returned by the LangSmith API
        
        ---
        📤 RETURNS
        ----------
        List[dict]
            A list of project dictionaries. The structure depends on `more_info`:
            
            **When `more_info=False` (simplified):**
            ```python
            [
                {
                    "name": "Chat-LangChain",
                    "project_id": "787d5165-f110-43ff-a3fb-66ea1a70c971",
                    "agent_deployment_id": "deployment-123"  # Only if available
                },
                ...
            ]
            ```
            
            **When `more_info=True` (full details):**
            Returns complete project objects with all fields from the LangSmith API,
            including metadata, settings, statistics, and nested structures.
        
        ---
        🧪 EXAMPLES
        ------------
        1️⃣ **List first 5 projects (simplified)**
        ```python
        projects = list_projects(limit="5")
        ```
        
        2️⃣ **Search for projects with "Chat" in the name**
        ```python
        projects = list_projects(project_name="Chat", limit="10")
        ```
        
        3️⃣ **Get full project details**
        ```python
        projects = list_projects(limit="3", more_info="true")
        ```
        
        4️⃣ **Find a specific project with full details**
        ```python
        projects = list_projects(project_name="MyProject", more_info="true", limit="1")
        ```
        
        ---
        🧠 NOTES FOR AGENTS
        --------------------
        - Use `more_info="false"` for quick project discovery and listing
        - Use `more_info="true"` when you need detailed project information
        - The `agent_deployment_id` field is automatically extracted from nested
        project data when available, making it easy to identify agent deployments
        - Projects are filtered to exclude reference projects by default
        - The function uses `name_contains` for filtering, so partial matches work
        """
        try:
            client = get_client_from_context(ctx)
            parsed_limit = int(limit) if limit else 5
            parsed_more_info = more_info.lower() == "true"
            return list_projects_tool(client, limit=parsed_limit, project_name=project_name, more_info=parsed_more_info)
        except Exception as e:
            return {"error": str(e)}

    # Register dataset tools
    @mcp.tool()
    def list_datasets(
        dataset_ids: Optional[str] = None,
        data_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_name_contains: Optional[str] = None,
        metadata: Optional[str] = None,
        limit: str = "20",
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch LangSmith datasets.

        Note: If no arguments are provided, all datasets will be returned.

        Args:
            dataset_ids (Optional[str]): Dataset IDs to filter by as JSON array string (e.g., '["id1", "id2"]') or single ID
            data_type (Optional[str]): Filter by dataset data type (e.g., 'chat', 'kv')
            dataset_name (Optional[str]): Filter by exact dataset name
            dataset_name_contains (Optional[str]): Filter by substring in dataset name
            metadata (Optional[str]): Filter by metadata as JSON object string (e.g., '{"key": "value"}')
            limit (str): Max number of datasets to return as string (default: "20")
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the datasets and metadata,
                            or an error message if the datasets cannot be retrieved
        """
        try:
            client = get_client_from_context(ctx)
            
            # Parse list strings (JSON arrays)
            parsed_dataset_ids = None
            if dataset_ids is not None:
                try:
                    parsed_dataset_ids = json.loads(dataset_ids) if dataset_ids.startswith("[") else [dataset_ids]
                except (json.JSONDecodeError, AttributeError):
                    parsed_dataset_ids = [dataset_ids] if dataset_ids else None
            
            # Parse metadata (JSON object)
            parsed_metadata = None
            if metadata is not None:
                try:
                    parsed_metadata = json.loads(metadata) if metadata.startswith("{") else None
                except (json.JSONDecodeError, AttributeError):
                    parsed_metadata = None
            
            parsed_limit = int(limit) if limit else 20
            
            return list_datasets_tool(
                client,
                dataset_ids=parsed_dataset_ids,
                data_type=data_type,
                dataset_name=dataset_name,
                dataset_name_contains=dataset_name_contains,
                metadata=parsed_metadata,
                limit=parsed_limit,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def list_examples(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        example_ids: Optional[str] = None,
        filter: Optional[str] = None,
        metadata: Optional[str] = None,
        splits: Optional[str] = None,
        inline_s3_urls: Optional[str] = None,
        include_attachments: Optional[str] = None,
        as_of: Optional[str] = None,
        limit: Optional[str] = None,
        offset: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Fetch examples from a LangSmith dataset with advanced filtering options.

        Note: Either dataset_id, dataset_name, or example_ids must be provided.
        If multiple are provided, they are used in order of precedence: example_ids, dataset_id, dataset_name.

        Args:
            dataset_id (Optional[str]): Dataset ID to retrieve examples from
            dataset_name (Optional[str]): Dataset name to retrieve examples from
            example_ids (Optional[str]): Specific example IDs as JSON array string (e.g., '["id1", "id2"]') or single ID
            limit (Optional[str]): Maximum number of examples to return as string (e.g., "10")
            offset (Optional[str]): Number of examples to skip as string (e.g., "0")
            filter (Optional[str]): Filter string using LangSmith query syntax (e.g., 'has(metadata, {"key": "value"})')
            metadata (Optional[str]): Metadata to filter by as JSON object string (e.g., '{"key": "value"}')
            splits (Optional[str]): Dataset splits as JSON array string (e.g., '["train", "test"]') or single split
            inline_s3_urls (Optional[str]): Whether to inline S3 URLs: "true" or "false" (default: SDK default if not specified)
            include_attachments (Optional[str]): Whether to include attachments: "true" or "false" (default: SDK default if not specified)
            as_of (Optional[str]): Dataset version tag OR ISO timestamp to retrieve examples as of that version/time
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the examples and metadata,
                            or an error message if the examples cannot be retrieved
        """
        try:
            client = get_client_from_context(ctx)
            
            # Parse list strings (JSON arrays)
            parsed_example_ids = None
            if example_ids is not None:
                try:
                    parsed_example_ids = json.loads(example_ids) if example_ids.startswith("[") else [example_ids]
                except (json.JSONDecodeError, AttributeError):
                    parsed_example_ids = [example_ids] if example_ids else None
            
            parsed_splits = None
            if splits is not None:
                try:
                    parsed_splits = json.loads(splits) if splits.startswith("[") else [splits]
                except (json.JSONDecodeError, AttributeError):
                    parsed_splits = [splits] if splits else None
            
            # Parse metadata (JSON object)
            parsed_metadata = None
            if metadata is not None:
                try:
                    parsed_metadata = json.loads(metadata) if metadata.startswith("{") else None
                except (json.JSONDecodeError, AttributeError):
                    parsed_metadata = None
            
            # Parse boolean strings
            parsed_inline_s3_urls = None
            if inline_s3_urls is not None:
                parsed_inline_s3_urls = inline_s3_urls.lower() == "true"
            
            parsed_include_attachments = None
            if include_attachments is not None:
                parsed_include_attachments = include_attachments.lower() == "true"
            
            # Parse integer strings
            parsed_limit = int(limit) if limit else None
            parsed_offset = int(offset) if offset else None
            
            return list_examples_tool(
                client,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                example_ids=parsed_example_ids,
                filter=filter,
                metadata=parsed_metadata,
                splits=parsed_splits,
                inline_s3_urls=parsed_inline_s3_urls,
                include_attachments=parsed_include_attachments,
                as_of=as_of,
                limit=parsed_limit,
                offset=parsed_offset,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def read_dataset(
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Read a specific dataset from LangSmith.

        Note: Either dataset_id or dataset_name must be provided to identify the dataset.
        If both are provided, dataset_id takes precedence.

        Args:
            dataset_id (Optional[str]): Dataset ID to retrieve
            dataset_name (Optional[str]): Dataset name to retrieve
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the dataset details,
                            or an error message if the dataset cannot be retrieved
        """
        try:
            client = get_client_from_context(ctx)
            return read_dataset_tool(
                client,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
            )
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def read_example(
        example_id: str,
        as_of: Optional[str] = None,
        ctx: Context = None,
    ) -> Dict[str, Any]:
        """
        Read a specific example from LangSmith.

        Args:
            example_id (str): Example ID to retrieve
            as_of (Optional[str]): Dataset version tag OR ISO timestamp to retrieve the example as of that version/time
            ctx: FastMCP context (automatically provided)

        Returns:
            Dict[str, Any]: Dictionary containing the example details,
                            or an error message if the example cannot be retrieved
        """
        try:
            client = get_client_from_context(ctx)
            return read_example_tool(
                client,
                example_id=example_id,
                as_of=as_of,
            )
        except Exception as e:
            return {"error": str(e)}
