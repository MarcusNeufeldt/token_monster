"""
Worker Agent module for Grok Heavy Multi-Agent AI System.
Handles individual agent execution with LLM integration and tool usage.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List
from tools import ToolExecutor, parse_tool_requests

logger = logging.getLogger(__name__)


class WorkerAgent:
    """Individual worker agent that processes tasks using a specific LLM."""
    
    def __init__(self, name: str, model: str, config: Dict[str, Any], tool_executor: ToolExecutor):
        self.name = name
        self.model = model
        self.config = config
        self.tool_executor = tool_executor
        self.api_config = config.get('api', {})
        self.base_url = self.api_config.get('base_url', 'https://openrouter.ai/api/v1')
        self.timeout = self.api_config.get('timeout', 60)
        self.max_retries = self.api_config.get('max_retries', 3)
        
        # Extract agent-specific config
        self.agent_config = self._find_agent_config()
        
    def _find_agent_config(self) -> Dict[str, Any]:
        """Find this agent's configuration in the config file."""
        agents = self.config.get('agents', [])
        for agent in agents:
            if agent.get('name') == self.name:
                return agent
        return {}
        
    async def process_task(self, prompt: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process a task and return the result with tool usage logs."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initial LLM call
            response = await self._call_llm(prompt, session)
            if not response['success']:
                return response
            
            # Check for tool usage and handle iteratively
            final_response = await self._handle_tool_usage(response['result'], session)
            
            end_time = asyncio.get_event_loop().time()
            
            return {
                'success': True,
                'agent_name': self.name,
                'model': self.model,
                'response': final_response['content'],
                'tool_usage': final_response.get('tool_usage', []),
                'tool_costs': final_response.get('tool_costs', {}),
                'processing_time': end_time - start_time,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Worker {self.name} failed: {str(e)}")
            return {
                'success': False,
                'agent_name': self.name,
                'model': self.model,
                'response': None,
                'tool_usage': [],
                'processing_time': 0,
                'error': str(e)
            }
    
    async def _handle_tool_usage(self, initial_response: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Handle tool usage in an iterative manner."""
        current_response = initial_response
        tool_usage_log = []
        tool_costs = []  # Track tool costs
        max_iterations = 5  # Prevent infinite loops
        
        for iteration in range(max_iterations):
            # Parse tool requests from current response
            tool_requests = parse_tool_requests(current_response)
            
            if not tool_requests:
                # No more tools needed
                break
                
            # Execute tools
            tool_results = []
            for tool_request in tool_requests:
                tool_name = tool_request['tool']
                tool_args = tool_request['args']
                
                result = await self.tool_executor.execute_tool(tool_name, **tool_args)
                tool_results.append({
                    'tool': tool_name,
                    'args': tool_args,
                    'result': result
                })
                
                # Collect tool cost information
                if 'cost' in result:
                    tool_costs.append({
                        'tool': tool_name,
                        'cost_info': result['cost']
                    })
                
                tool_usage_log.append({
                    'iteration': iteration,
                    'tool': tool_name,
                    'args': tool_args,
                    'success': result['success'],
                    'result': result['result'],
                    'error': result.get('error'),
                    'cost': result.get('cost', {'cost': 0.0})
                })
            
            # Format tool results for the LLM
            tool_results_text = self._format_tool_results(tool_results)
            
            # Create follow-up prompt with tool results
            follow_up_prompt = f"""
Previous response: {current_response}

Tool results:
{tool_results_text}

Please provide your final response incorporating the tool results. Do not use any more tools unless absolutely necessary.
"""
            
            # Make follow-up LLM call
            follow_up_response = await self._call_llm(follow_up_prompt, session)
            if follow_up_response['success']:
                current_response = follow_up_response['result']
            else:
                logger.warning(f"Follow-up call failed for {self.name}: {follow_up_response['error']}")
                break
        
        # Calculate total tool costs
        total_tool_cost = self._calculate_total_tool_cost(tool_costs)
        
        return {
            'content': current_response,
            'tool_usage': tool_usage_log,
            'tool_costs': total_tool_cost
        }
    
    def _format_tool_results(self, tool_results: List[Dict[str, Any]]) -> str:
        """Format tool results for inclusion in LLM prompt."""
        formatted_results = []
        
        for result in tool_results:
            tool_name = result['tool']
            tool_result = result['result']
            
            if tool_result['success']:
                if tool_name == 'web_search':
                    search_results = tool_result['result']['results']
                    formatted = f"Web search results for '{tool_result['result']['query']}':\n"
                    for i, res in enumerate(search_results[:3], 1):  # Limit to top 3
                        formatted += f"{i}. {res['title']}: {res['body'][:200]}...\n"
                elif tool_name == 'code_executor':
                    stdout = tool_result['result']['stdout']
                    stderr = tool_result['result']['stderr']
                    formatted = f"Code execution result:\n"
                    if stdout:
                        formatted += f"Output: {stdout}\n"
                    if stderr:
                        formatted += f"Errors: {stderr}\n"
                else:
                    formatted = f"Tool {tool_name} result: {str(tool_result['result'])}"
            else:
                formatted = f"Tool {tool_name} failed: {tool_result['error']}"
            
            formatted_results.append(formatted)
        
        return '\n\n'.join(formatted_results)
    
    async def _call_llm(self, prompt: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Make an API call to the LLM with retry logic and self-healing."""
        headers = {
            'Authorization': f'Bearer {self.config.get("api_key")}',
            'Content-Type': 'application/json'
        }
        
        # Prepare the payload
        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': self._get_system_prompt()
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 4000,  # Increased for Grok 3 mini reasoning model
            'usage': {
                'include': True  # Enable usage tracking
            }
        }
        
        # Add reasoning effort for supported models
        if self.agent_config.get('reasoning_effort'):
            payload['reasoning'] = {'effort': self.agent_config['reasoning_effort']}
        
        url = f"{self.base_url}/chat/completions"
        
        # Self-healing: Track failures and adapt
        consecutive_failures = 0
        
        for attempt in range(self.max_retries):
            try:
                # Self-healing: Reduce max_tokens if we've had failures (but keep reasonable minimum for reasoning)
                if consecutive_failures > 0:
                    payload['max_tokens'] = max(1000, payload['max_tokens'] - (consecutive_failures * 500))
                    logger.info(f"{self.name}: Self-healing - reducing max_tokens to {payload['max_tokens']}")
                
                async with session.post(
                    url, 
                    headers=headers, 
                    json=payload, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Extract usage statistics
                        usage_stats = data.get('usage', {})
                        
                        # Self-healing: Reset failure counter on success
                        if consecutive_failures > 0:
                            logger.info(f"{self.name}: Self-healing successful after {consecutive_failures} failures")
                        
                        # Log usage statistics
                        if usage_stats:
                            logger.info(f"{self.name}: Usage - Input: {usage_stats.get('prompt_tokens', 0)} tokens, "
                                      f"Output: {usage_stats.get('completion_tokens', 0)} tokens, "
                                      f"Total: {usage_stats.get('total_tokens', 0)} tokens")
                        
                        return {
                            'success': True,
                            'result': content,
                            'error': None,
                            'self_healing_applied': consecutive_failures > 0,
                            'final_max_tokens': payload['max_tokens'],
                            'usage_stats': usage_stats
                        }
                    else:
                        consecutive_failures += 1
                        error_text = await response.text()
                        logger.warning(f"API call failed for {self.name}, attempt {attempt + 1}: {response.status} - {error_text}")
                        
                        # Self-healing: Try different approach based on error type
                        if response.status == 429:  # Rate limit
                            await asyncio.sleep(2 ** attempt + consecutive_failures)
                        elif response.status >= 500:  # Server error
                            payload['temperature'] = max(0.1, payload['temperature'] - 0.1)  # Reduce randomness
                            logger.info(f"{self.name}: Self-healing - reducing temperature to {payload['temperature']}")
                        
                        if attempt == self.max_retries - 1:
                            return {
                                'success': False,
                                'result': None,
                                'error': f"API call failed after {self.max_retries} attempts: {response.status} - {error_text}",
                                'self_healing_attempted': True,
                                'final_attempt_params': payload
                            }
                        
            except asyncio.TimeoutError:
                consecutive_failures += 1
                logger.warning(f"Timeout for {self.name}, attempt {attempt + 1}")
                
                # Self-healing: Reduce timeout for subsequent attempts
                self.timeout = max(30, self.timeout - 10)
                logger.info(f"{self.name}: Self-healing - reducing timeout to {self.timeout}s")
                
                if attempt == self.max_retries - 1:
                    return {
                        'success': False,
                        'result': None,
                        'error': f"Request timed out after {self.max_retries} attempts",
                        'self_healing_attempted': True
                    }
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"API call error for {self.name}, attempt {attempt + 1}: {str(e)}")
                
                # Self-healing: Simplify request on unknown errors
                if 'reasoning' in payload and consecutive_failures > 1:
                    del payload['reasoning']
                    logger.info(f"{self.name}: Self-healing - removing reasoning parameter")
                
                if attempt == self.max_retries - 1:
                    return {
                        'success': False,
                        'result': None,
                        'error': str(e),
                        'self_healing_attempted': True
                    }
                await asyncio.sleep(2 ** attempt)
        
        return {
            'success': False,
            'result': None,
            'error': "Max retries exceeded",
            'self_healing_attempted': True
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent based on their specialized role."""
        agent_role = self.agent_config.get('role', 'generalist')
        agent_specialization = self.agent_config.get('specialization', 'general')
        agent_capabilities = self.agent_config.get('capabilities', [])
        
        # Base system prompt components
        base_intro = f"""You are {self.name}, a specialized AI agent using the {self.model} model. 
You are part of a multi-agent system designed to provide high-quality, accurate responses through collaborative deliberation."""
        
        # Role-specific system prompts
        if agent_role == 'planner':
            role_prompt = f"""
YOUR ROLE: Strategic Planner & Task Decomposition Specialist

Your primary responsibilities:
1. Analyze complex queries and break them down into clear, actionable sub-tasks
2. Identify what information is needed and how to obtain it
3. Determine the optimal sequence of operations
4. Consider resource requirements and potential challenges
5. Create structured plans that other agents can follow

Your approach should be:
- Systematic and methodical in breaking down problems
- Clear in defining objectives and success criteria
- Strategic in considering multiple approaches
- Comprehensive in identifying all necessary steps
- Practical in ensuring plans are executable

Focus on creating detailed, actionable plans rather than executing the tasks yourself."""

        elif agent_role == 'researcher':
            role_prompt = f"""
YOUR ROLE: Research Specialist & Information Gatherer

Your primary responsibilities:
1. Execute research tasks using available tools (web search, code execution, data analysis)
2. Gather comprehensive, accurate information from multiple sources
3. Verify facts and cross-reference information
4. Organize findings in a clear, structured manner
5. Focus on depth and accuracy over speed

Your approach should be:
- Thorough in investigating all relevant sources
- Critical in evaluating information quality
- Systematic in organizing findings
- Tool-savvy in leveraging web search, code execution, and data analysis
- Fact-focused rather than opinion-based

Use tools strategically based on the guidelines provided. Prioritize primary sources and current information."""

        elif agent_role == 'critic':
            role_prompt = f"""
YOUR ROLE: Quality Assurance & Critical Analysis Specialist

Your primary responsibilities:
1. Review work from other agents for accuracy, completeness, and quality
2. Identify potential errors, biases, or missing information
3. Evaluate logical consistency and argument strength
4. Check for factual accuracy and source reliability
5. Suggest specific improvements and corrections

Your approach should be:
- Rigorous in examining details and logic
- Objective in identifying weaknesses without personal bias
- Constructive in providing actionable feedback
- Thorough in checking all claims and assertions
- Balanced in recognizing both strengths and weaknesses

Be critical but fair. Focus on improving the overall quality of the response."""

        elif agent_role == 'synthesizer':
            role_prompt = f"""
YOUR ROLE: Content Creation & Synthesis Specialist

Your primary responsibilities:
1. Combine research and analysis into coherent, comprehensive responses
2. Structure information in a logical, easy-to-follow format
3. Ensure clarity, readability, and appropriate depth
4. Address the original query completely and directly
5. Create polished, professional final outputs

Your approach should be:
- Clear and organized in presentation
- Comprehensive in addressing all aspects of the query
- Engaging and readable in writing style
- Accurate in representing research findings
- Complete in providing actionable information

Focus on creating responses that are both informative and accessible to the user."""

        else:
            # Fallback for generalist or unknown roles
            role_prompt = f"""
YOUR ROLE: General AI Assistant

Your responsibilities:
1. Provide thoughtful, accurate, and detailed responses to user queries
2. Use your built-in knowledge for factual information and well-established topics
3. Only use tools when they provide genuine value that your training data cannot
4. Be transparent about your reasoning process
5. Collaborate effectively with other agents"""

        # Tool usage guidelines (consistent across all roles)
        tool_guidelines = f"""
Available tools:
- <tool name="web_search" query="search terms"/>: Search the web for current information
- <tool name="code_executor" code="python code"/>: Execute Python code safely
- <tool name="data_explorer" query="analysis request">CSV or JSON data</tool>: Analyze data with pandas

IMPORTANT - Tool usage guidelines:
- Use web_search ONLY for: current events, recent news, real-time data, specific recent information, or when you need to verify very recent changes
- Do NOT use web_search for: basic facts, historical information, general knowledge, definitions, or well-established information
- Use code_executor for: calculations, data processing, algorithm demonstrations, or when computational results are needed
- Use data_explorer for: analyzing provided datasets, CSV/JSON data processing
- For basic factual questions (like "What is the capital of Spain?"), rely on your training data
- Be specific and targeted in your tool usage when tools are genuinely needed
- Format tool calls exactly as shown above
- For data_explorer, put CSV/JSON data between opening and closing tags

Examples of when NOT to use tools:
- "What is the capital of Spain?" → Use built-in knowledge
- "Explain photosynthesis" → Use built-in knowledge
- "What is 2+2?" → Use built-in knowledge
- "How does machine learning work?" → Use built-in knowledge

Examples of when TO use tools:
- "What's the latest news about AI?" → Use web_search
- "What's the current stock price of Tesla?" → Use web_search
- "Calculate the compound interest for..." → Use code_executor
- "Analyze this sales data: [CSV data]" → Use data_explorer"""

        return f"""{base_intro}

{role_prompt}

{tool_guidelines}

Remember: Your specialized role is {agent_role} with focus on {agent_specialization}. 
Leverage your capabilities: {', '.join(agent_capabilities)} to excel in your domain while collaborating effectively with other specialized agents.

Provide comprehensive, well-reasoned responses that demonstrate your unique perspective and specialized capabilities.""" 

    def _calculate_total_tool_cost(self, tool_costs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate total cost from all tools used by this agent."""
        total_cost = 0.0
        breakdown = {
            'exa_searches': 0,
            'code_executions': 0,
            'data_analyses': 0,
            'costs': {
                'exa_cost': 0.0,
                'code_cost': 0.0,  # Always 0.0
                'data_cost': 0.0,  # Always 0.0
                'total_tool_cost': 0.0
            }
        }
        
        for tool_cost in tool_costs:
            tool_name = tool_cost['tool']
            cost_info = tool_cost['cost_info']
            cost = cost_info.get('cost', 0.0)
            total_cost += cost
            
            if tool_name == 'web_search':
                breakdown['exa_searches'] += cost_info.get('exa_searches', 0)
                breakdown['costs']['exa_cost'] += cost
            elif tool_name == 'code_executor':
                breakdown['code_executions'] += cost_info.get('executions', 0)
                breakdown['costs']['code_cost'] += cost
            elif tool_name == 'data_explorer':
                breakdown['data_analyses'] += cost_info.get('analyses', 0)
                breakdown['costs']['data_cost'] += cost
        
        breakdown['costs']['total_tool_cost'] = total_cost
        
        return breakdown 