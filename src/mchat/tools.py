import asyncio
import inspect
import json
from pathlib import Path
from typing import get_type_hints
from urllib.parse import quote

import aiofiles
import httpx
from bs4 import BeautifulSoup

from mchat.config import get_config


async def web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search using Google engine.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted string containing search results with title, URL, and snippet
        for each result, separated by "---" dividers
    """
    config = get_config()

    url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?key={config.google_api_key}"
        f"&cx={config.google_search_engine_id}"
        f"&q={quote(query)}"
    )

    async with httpx.AsyncClient() as client:
        results = []
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        items = data["items"][:max_results]
        for item in items:
            results.append(
                f"Title: {item['title']}\nURL: {item['link']}\nSnippet: {item['snippet']}...\n---"
            )
        return "\n".join(results)


async def extract_web_page(url: str) -> str:
    """
    Extract text content from a web page given its URL

    Args:
        url: The URL of web page to extract text from

    Returns:
        The extracted text content from the web page
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)


async def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read content from a file.

    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)

    Returns:
        str: File content, or error message starting with "Error:"
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    async with aiofiles.open(path, "r", encoding=encoding) as f:
        content = await f.read()
    return content


async def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file.

    Args:
        file_path: Path where to write the file
        content: Content to write
        encoding: File encoding (default: utf-8)

    Returns:
        str: Success message or error message starting with "Error:"
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w", encoding=encoding) as f:
        await f.write(content)
    return f"Successfully wrote {len(content)} characters to {file_path}"


_TOOLS = {
    "extract_web_page": extract_web_page,
    "web_search": web_search,
    "read_file": read_file,
    "write_file": write_file,
}


def _get_tool_param_type(param_name, type_hints):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_map.get(type_hints.get(param_name, str))


def get_tool_schemas():
    schemas = []
    for tool in _TOOLS.values():
        sig = inspect.signature(tool)
        type_hints = get_type_hints(tool)

        doc = tool.__doc__ or ""
        lines = [line.strip() for line in doc.split("\n") if line.strip()]

        description = lines[0] if lines else tool.__name__

        properties = {}
        required = []

        in_args = False
        for line in lines:
            if line.startswith("Args:"):
                in_args = True
            elif line.startswith("Returns:"):
                break
            elif in_args and ":" in line:
                param_name = line.split(":")[0].strip()
                param_desc = line.split(":", 1)[1].strip()

                param = sig.parameters.get(param_name)
                if param:
                    properties[param_name] = {
                        "type": _get_tool_param_type(param_name, type_hints),
                        "description": param_desc,
                    }

                    if param.default == inspect.Parameter.empty:
                        required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        schemas.append(schema)
    return schemas


async def exec_tool_calls(tool_calls: list[dict]) -> list[dict]:
    tasks = [_execute_single_tool(tool_call) for tool_call in tool_calls]
    results = await asyncio.gather(*tasks)

    return results


async def _execute_single_tool(tool_call: dict) -> dict:
    try:
        if tool_call["type"] != "function":
            return {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": "Error: Invalid tool call type",
            }

        fn_name = tool_call["function"]["name"]
        args_json = tool_call["function"]["arguments"]

        if fn_name not in _TOOLS:
            return {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": f"Error: Unknown function '{fn_name}'",
            }

        fn = _TOOLS[fn_name]
        args = json.loads(args_json)
        result = await fn(**args)

        return {"role": "tool", "tool_call_id": tool_call["id"], "content": result}

    except json.JSONDecodeError as e:
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": f"Error: Invalid JSON arguments - {str(e)}",
        }
    except Exception as e:
        return {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": f"Error: {str(e)}",
        }
