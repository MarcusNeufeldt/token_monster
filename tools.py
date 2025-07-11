"""
Tools module for Grok Heavy Multi-Agent AI System.
Provides web search and code execution capabilities for agents.
"""

import asyncio
import subprocess
import tempfile
import os
import sys
from typing import Dict, Any, Optional
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Handles execution of various tools that agents can use."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools_config = config.get('tools', {})
        
        # Initialize Exa client
        exa_api_key = os.getenv('EXA_API_KEY')
        if exa_api_key:
            self.exa_client = OpenAI(
                base_url="https://api.exa.ai",
                api_key=exa_api_key
            )
        else:
            self.exa_client = None
            logger.warning("EXA_API_KEY not found in environment variables")
        
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        try:
            if tool_name == "web_search":
                return await self.web_search(kwargs.get('query', ''))
            elif tool_name == "code_executor":
                return await self.execute_code(kwargs.get('code', ''), kwargs.get('output_dir'))
            elif tool_name == "data_explorer":
                return await self.explore_data(kwargs.get('data', ''), kwargs.get('query', ''))
            else:
                return {
                    "success": False,
                    "error": f"Unknown tool: {tool_name}",
                    "result": None
                }
        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": None
            }
    
    async def web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search using Exa."""
        if not self.tools_config.get('web_search', {}).get('enabled', True):
            return {
                "success": False,
                "error": "Web search is disabled",
                "result": None,
                "cost": {"exa_searches": 0, "cost": 0.0}
            }
        
        if not query.strip():
            return {
                "success": False,
                "error": "Empty search query",
                "result": None,
                "cost": {"exa_searches": 0, "cost": 0.0}
            }
        
        if not self.exa_client:
            return {
                "success": False,
                "error": "Exa client not initialized - check EXA_API_KEY",
                "result": None,
                "cost": {"exa_searches": 0, "cost": 0.0}
            }
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._search_exa, 
                query
            )
            
            # Calculate Exa cost: $5.00 per 1k answers
            # Each search counts as 1 answer
            exa_cost = 0.005  # $5.00 / 1000 = $0.005 per search
            
            return {
                "success": True,
                "error": None,
                "result": {
                    "query": query,
                    "results": results
                },
                "cost": {
                    "exa_searches": 1,
                    "cost": exa_cost,
                    "currency": "USD",
                    "pricing_note": "$5.00 per 1k answers"
                }
            }
            
        except Exception as e:
            logger.error(f"Exa search error: {str(e)}")
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "result": None,
                "cost": {"exa_searches": 0, "cost": 0.0}
            }
    
    def _search_exa(self, query: str) -> list:
        """Helper method to perform Exa search."""
        try:
            # Use Exa's chat completion API for search
            completion = self.exa_client.chat.completions.create(
                model="exa",
                messages=[{"role": "user", "content": f"Search for: {query}"}],
                stream=False
            )
            
            # Parse the response
            response_content = completion.choices[0].message.content
            
            # For now, return the response as a single result
            # In a real implementation, you might want to parse this differently
            # based on Exa's actual response format
            results = [{
                "title": f"Exa Search Results for: {query}",
                "body": response_content,
                "href": "https://exa.ai"
            }]
            
            return results
            
        except Exception as e:
            logger.error(f"Exa search error: {str(e)}")
            return []
    
    async def execute_code(self, code: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute Python code in a sandboxed environment."""
        if not self.tools_config.get('code_executor', {}).get('enabled', True):
            return {
                "success": False,
                "error": "Code execution is disabled",
                "result": None,
                "cost": {"executions": 0, "cost": 0.0}
            }
        
        if not code.strip():
            return {
                "success": False,
                "error": "Empty code",
                "result": None,
                "cost": {"executions": 0, "cost": 0.0}
            }
        
        try:
            timeout = self.tools_config.get('code_executor', {}).get('timeout', 10)
            
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Set working directory - use output_dir if provided, otherwise use temp directory
                working_dir = output_dir if output_dir else tempfile.gettempdir()
                
                # Execute the code in a subprocess for sandboxing
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir
                )
                
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "error": stderr.decode() if stderr else None,
                    "result": {
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                        "return_code": process.returncode
                    },
                    "cost": {
                        "executions": 1,
                        "cost": 0.0,
                        "currency": "USD",
                        "pricing_note": "Free local execution"
                    }
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": f"Code execution timed out after {timeout} seconds",
                "result": None,
                "cost": {"executions": 0, "cost": 0.0}
            }
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "cost": {"executions": 0, "cost": 0.0}
            }
    
    async def explore_data(self, data: str, query: str) -> Dict[str, Any]:
        """Explore and analyze data using pandas in a sandboxed environment."""
        if not self.tools_config.get('data_explorer', {}).get('enabled', True):
            return {
                "success": False,
                "error": "Data exploration is disabled",
                "result": None,
                "cost": {"analyses": 0, "cost": 0.0}
            }
        
        if not data.strip() or not query.strip():
            return {
                "success": False,
                "error": "Empty data or query",
                "result": None,
                "cost": {"analyses": 0, "cost": 0.0}
            }
        
        try:
            # Create analysis code
            analysis_code = f'''
import pandas as pd
import numpy as np
import json
from io import StringIO

# Load data
try:
    # Try to parse as CSV first
    data_str = """{data}"""
    df = pd.read_csv(StringIO(data_str))
except:
    try:
        # Try to parse as JSON
        import json
        data_json = json.loads(data_str)
        df = pd.DataFrame(data_json)
    except:
        print("Error: Could not parse data as CSV or JSON")
        exit(1)

print("Data loaded successfully!")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print("\\nFirst few rows:")
print(df.head())

# Analysis based on query
query = "{query}"
print(f"\\nAnalysis for: {{query}}")

# Basic statistics
if df.select_dtypes(include=[np.number]).shape[1] > 0:
    print("\\nNumeric columns summary:")
    print(df.describe())

# Query-specific analysis
query_lower = query.lower()
if "trend" in query_lower or "pattern" in query_lower:
    print("\\nTrend analysis:")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"{{col}}: mean={{df[col].mean():.2f}}, std={{df[col].std():.2f}}")

if "correlation" in query_lower:
    print("\\nCorrelation matrix:")
    print(df.corr())

if "missing" in query_lower or "null" in query_lower:
    print("\\nMissing values:")
    print(df.isnull().sum())

if "unique" in query_lower:
    print("\\nUnique values per column:")
    for col in df.columns:
        print(f"{{col}}: {{df[col].nunique()}} unique values")
'''
            
            result = await self.execute_code(analysis_code)
            
            # Add data exploration specific cost info
            if result["success"]:
                result["cost"] = {
                    "analyses": 1,
                    "cost": 0.0,
                    "currency": "USD",
                    "pricing_note": "Free local pandas analysis"
                }
            else:
                result["cost"] = {"analyses": 0, "cost": 0.0}
            
            return result
            
        except Exception as e:
            logger.error(f"Data exploration error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "cost": {"analyses": 0, "cost": 0.0}
            }


def parse_tool_requests(text: str) -> list:
    """Parse tool requests from agent output."""
    import re
    
    tool_requests = []
    
    # Look for <tool>toolname</tool> patterns
    tool_pattern = r'<tool>(\w+)</tool>'
    matches = re.findall(tool_pattern, text)
    
    for match in matches:
        tool_requests.append({
            "tool": match,
            "args": {}
        })
    
    # Look for more structured tool calls like <tool name="web_search" query="example"/>
    structured_pattern = r'<tool\s+name="([^"]+)"([^>]*)/>'
    structured_matches = re.findall(structured_pattern, text)
    
    for tool_name, args_str in structured_matches:
        args = {}
        # Parse arguments from the args_str
        arg_pattern = r'(\w+)="([^"]*)"'
        arg_matches = re.findall(arg_pattern, args_str)
        for arg_name, arg_value in arg_matches:
            if arg_name != "name":  # Skip the name attribute
                args[arg_name] = arg_value
        
        tool_requests.append({
            "tool": tool_name,
            "args": args
        })
    
    # Look for data explorer pattern with multi-line data
    data_explorer_pattern = r'<tool name="data_explorer" query="([^"]*)">\s*(.*?)\s*</tool>'
    data_matches = re.findall(data_explorer_pattern, text, re.DOTALL)
    
    for query, data in data_matches:
        tool_requests.append({
            "tool": "data_explorer",
            "args": {
                "query": query,
                "data": data.strip()
            }
        })
    
    return tool_requests 