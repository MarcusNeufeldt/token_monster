"""
Supervisor Agent module for Grok Heavy Multi-Agent AI System.
Orchestrates the multi-agent workflow and handles deliberation.
"""

import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from worker import WorkerAgent
from tools import ToolExecutor

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Central orchestrator that manages the multi-agent workflow."""
    
    def __init__(self, config: Dict[str, Any], api_key: str):
        self.config = config
        self.api_key = api_key
        self.deliberation_config = config.get('deliberation', {})
        self.system_config = config.get('system', {})
        
        # Create base worker agents
        self.workers = self._create_workers()
        self.spawned_agents = []  # Track dynamically spawned agents
        
        # Setup logging directory
        logs_dir = self.system_config.get('logs_directory', 'logs')
        self.logs_dir = logs_dir
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

    def _spawn_agent(self, template_name: str, focus_area: str, reason: str) -> Optional[WorkerAgent]:
        """Spawn a new specialized agent on-the-fly based on a template."""
        spawning_config = self.config.get('dynamic_spawning', {})
        if not spawning_config.get('enabled', False):
            return None
        
        templates = spawning_config.get('agent_templates', {})
        if template_name not in templates:
            logger.warning(f"Template '{template_name}' not found in agent templates")
            return None
        
        template = templates[template_name]
        
        # Create unique name for spawned agent
        spawned_count = len([a for a in self.spawned_agents if hasattr(a, 'agent_config') and a.agent_config.get('template') == template_name])
        agent_name = f"{template_name}_{spawned_count + 1}_{focus_area.replace(' ', '_')[:10]}"
        
        # Create agent configuration
        agent_config = {
            'name': agent_name,
            'model': template.get('model', self.workers[0].model if self.workers else 'x-ai/grok-3-mini-beta'),
            'role': template.get('base_role', 'researcher'),
            'specialization': template.get('specialization', 'specialized_research'),
            'focus_area': focus_area,
            'capabilities': template.get('capabilities', ['research']),
            'template': template_name,
            'spawned_for': reason,
            'reasoning_effort': template.get('reasoning_effort', 'high')
        }
        
        # Import WorkerAgent and ToolExecutor
        from worker import WorkerAgent
        from tools import ToolExecutor
        
        # Create tool executor
        tool_executor = ToolExecutor(self.config)
        
        # Prepare config with API key for workers
        worker_config = self.config.copy()
        worker_config['api_key'] = self.api_key
        
        # Create the spawned agent
        spawned_agent = WorkerAgent(agent_name, agent_config['model'], worker_config, tool_executor)
        
        # Set the agent config as an attribute to ensure it's accessible
        spawned_agent.agent_config = agent_config
        
        # For existing workers that might not have agent_config, create default ones
        for worker in self.workers:
            if not hasattr(worker, 'agent_config'):
                # Find the worker's config from the original config
                worker_config_from_file = None
                for agent_config_item in self.config.get('agents', []):
                    if agent_config_item.get('name') == worker.name:
                        worker_config_from_file = agent_config_item
                        break
                
                if worker_config_from_file:
                    worker.agent_config = worker_config_from_file
                else:
                    # Create default config for worker
                    worker.agent_config = {
                        'name': worker.name,
                        'model': worker.model,
                        'role': 'researcher',  # Default role
                        'specialization': 'general_research',
                        'focus_area': 'general',
                        'capabilities': ['research'],
                        'template': 'base_worker',
                        'spawned_for': 'base configuration',
                        'reasoning_effort': 'high'
                    }
        
        # Add to spawned agents list
        self.spawned_agents.append(spawned_agent)
        
        logger.info(f"Spawned specialized agent: {agent_name} for {reason}")
        
        return spawned_agent

    def _get_active_agents(self, complexity_analysis: Dict[str, Any]) -> List[WorkerAgent]:
        """Get the list of active agents including any dynamically spawned ones."""
        active_agents = self.workers.copy()
        
        # Spawn agents if needed
        if complexity_analysis.get('should_spawn_agents', False):
            spawning_recommendations = complexity_analysis.get('spawning_recommendations', [])
            max_spawned = self.config.get('dynamic_spawning', {}).get('max_spawned_agents', 6)
            
            for recommendation in spawning_recommendations:
                if len(active_agents) >= max_spawned:
                    logger.warning(f"Reached maximum agent limit ({max_spawned}), skipping further spawning")
                    break
                
                spawned_agent = self._spawn_agent(
                    recommendation['template'],
                    recommendation['focus_area'],
                    recommendation['reason']
                )
                
                if spawned_agent:
                    active_agents.append(spawned_agent)
        
        return active_agents

    def _cleanup_spawned_agents(self):
        """Clean up spawned agents after query completion."""
        if self.spawned_agents:
            logger.info(f"Cleaning up {len(self.spawned_agents)} spawned agents")
            self.spawned_agents.clear()
    
    def _create_workers(self) -> List[WorkerAgent]:
        """Create worker agents from configuration."""
        from worker import WorkerAgent
        from tools import ToolExecutor
        
        workers = []
        agents_config = self.config.get('agents', [])
        
        # Create tool executor once for all workers
        tool_executor = ToolExecutor(self.config)
        
        # Prepare config with API key for workers
        worker_config = self.config.copy()
        worker_config['api_key'] = self.api_key
        
        for agent_config in agents_config:
            name = agent_config.get('name', f'agent_{len(workers) + 1}')
            model = agent_config.get('model', 'x-ai/grok-3-mini-beta')
            
            worker = WorkerAgent(name, model, worker_config, tool_executor)
            workers.append(worker)
            
            logger.info(f"Created worker agent: {name} ({model})")
        
        return workers
    
    async def process_query(self, query: str, verbose: bool = False, force_agent_count: Optional[int] = None) -> Dict[str, Any]:
        """Process a user query through the specialized multi-agent orchestration system with dynamic spawning."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = asyncio.get_event_loop().time()
        
        logger.info(f"Processing query [Session: {session_id}]: {query[:100]}...")
        
        # Analyze query complexity and spawning needs
        complexity_analysis = self._analyze_query_complexity(query)
        
        # Get active agents (base + any spawned agents)
        active_agents = self._get_active_agents(complexity_analysis)
        
        if complexity_analysis.get('should_spawn_agents', False):
            spawned_count = len(self.spawned_agents)
            logger.info(f"Dynamic spawning activated: {spawned_count} specialized agents spawned")
            for agent in self.spawned_agents:
                logger.info(f"  - {agent.name}: {agent.agent_config.get('spawned_for', 'Unknown reason')}")
        
        # Check orchestration mode
        orchestration_mode = self.deliberation_config.get('orchestration_mode', 'parallel')
        enable_critique = self.deliberation_config.get('enable_critique_rounds', True)
        
        try:
            if orchestration_mode == 'sequential':
                # Sequential orchestration with potential spawned agents
                orchestration_result = await self._orchestrate_sequential_dynamic(query, session_id, enable_critique, active_agents)
            else:
                # Enhanced parallel orchestration with spawned agents
                orchestration_result = await self._orchestrate_parallel_dynamic(query, session_id, active_agents)
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # Compile final result
            result = {
                'success': orchestration_result['success'],
                'final_response': orchestration_result.get('final_response'),
                'error': orchestration_result.get('error'),
                'session_id': session_id,
                'orchestration_log': orchestration_result.get('orchestration_log', {}),
                'processing_time': processing_time,
                'query': query,
                'orchestration_mode': orchestration_mode,
                'complexity_analysis': complexity_analysis,
                'total_usage': orchestration_result.get('total_usage', {}),
                'dynamic_spawning': {
                    'enabled': complexity_analysis.get('spawning_enabled', False),
                    'spawned_count': len(self.spawned_agents),
                    'spawned_agents': [
                        {
                            'name': agent.name,
                            'specialization': agent.agent_config.get('specialization', 'unknown'),
                            'focus_area': agent.agent_config.get('focus_area', 'unknown'),
                            'reason': agent.agent_config.get('spawned_for', 'unknown')
                        }
                        for agent in self.spawned_agents
                    ]
                }
            }
            
            # Clean up spawned agents
            self._cleanup_spawned_agents()
            
            # Save logs
            if self.system_config.get('save_logs', True):
                await self._save_session_log(result)
            
            # Display results
            self._display_results_enhanced(result, verbose)
            
            return result
            
        except Exception as e:
            logger.error(f"Supervisor processing error: {str(e)}")
            # Clean up spawned agents on error
            self._cleanup_spawned_agents()
            return {
                'success': False,
                'final_response': None,
                'error': str(e),
                'session_id': session_id,
                'orchestration_log': {},
                'processing_time': asyncio.get_event_loop().time() - start_time,
                'complexity_analysis': complexity_analysis,
                'dynamic_spawning': {
                    'enabled': complexity_analysis.get('spawning_enabled', False),
                    'spawned_count': 0,
                    'spawned_agents': []
                }
            }

    async def _orchestrate_sequential(self, query: str, session_id: str, enable_critique: bool = True) -> Dict[str, Any]:
        """Orchestrate agents in sequential specialized workflow: Planner -> Researcher -> Critic -> Synthesizer."""
        logger.info("Starting sequential orchestration with specialized agents...")
        
        orchestration_log = {
            'mode': 'sequential',
            'stages': [],
            'critique_rounds': 0,
            'agents_used': []
        }
        
        # Find specialized agents
        planner = self._find_agent_by_role('planner')
        researcher = self._find_agent_by_role('researcher')
        critic = self._find_agent_by_role('critic')
        synthesizer = self._find_agent_by_role('synthesizer')
        
        if not all([planner, researcher, critic, synthesizer]):
            logger.error("Not all required specialized agents found")
            return {
                'success': False,
                'error': 'Missing specialized agents (planner, researcher, critic, synthesizer)',
                'orchestration_log': orchestration_log
            }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Stage 1: Planning
                logger.info("Stage 1: Planning phase...")
                planning_prompt = f"""
Original Query: {query}

As the Strategic Planner, create a structured action plan for this query. Do NOT use any tools - focus only on planning.

Your task:
1. Break down this query into clear, actionable sub-tasks
2. Identify what information needs to be gathered
3. Recommend which tools the Researcher should use
4. Anticipate potential challenges

Provide a concise action plan with:
- Research objectives (3-5 key points to investigate)
- Recommended tools for each objective (web_search/code_executor/data_explorer)
- Expected deliverables
- Success criteria

Keep your response focused and actionable. Do not conduct research yourself - just plan it.
"""
                
                planning_result = await self._execute_agent_stage(planner, planning_prompt, session, "Planning")
                orchestration_log['stages'].append(planning_result)
                orchestration_log['agents_used'].append(planner.name)
                
                if not planning_result['success']:
                    return {'success': False, 'error': 'Planning stage failed', 'orchestration_log': orchestration_log}
                
                # Stage 2: Research
                logger.info("Stage 2: Research phase...")
                research_prompt = f"""
Original Query: {query}

Planning Results:
{planning_result['response']}

As the Research Specialist, your task is to:
1. Execute the research plan provided by the Planner
2. Use appropriate tools (web_search, code_executor, data_explorer) to gather information
3. Focus on accuracy, completeness, and relevance
4. Organize your findings clearly
5. Provide evidence and sources for your information

Follow the plan but use your expertise to adapt as needed. Be thorough and systematic in your research approach.
"""
                
                research_result = await self._execute_agent_stage(researcher, research_prompt, session, "Research")
                orchestration_log['stages'].append(research_result)
                orchestration_log['agents_used'].append(researcher.name)
                
                if not research_result['success']:
                    return {'success': False, 'error': 'Research stage failed', 'orchestration_log': orchestration_log}
                
                # Stage 3: Initial Synthesis
                logger.info("Stage 3: Initial synthesis...")
                synthesis_prompt = f"""
Original Query: {query}

Planning Results:
{planning_result['response']}

Research Results:
{research_result['response']}

As the Content Synthesizer, your task is to:
1. Combine the planning insights and research findings into a comprehensive response
2. Structure the information logically and clearly
3. Ensure the response directly addresses the original query
4. Make the content accessible and well-organized
5. Create a complete draft response

This is your initial synthesis. Focus on creating a comprehensive, well-structured response that incorporates all the research findings.
"""
                
                synthesis_result = await self._execute_agent_stage(synthesizer, synthesis_prompt, session, "Initial Synthesis")
                orchestration_log['stages'].append(synthesis_result)
                orchestration_log['agents_used'].append(synthesizer.name)
                
                if not synthesis_result['success']:
                    return {'success': False, 'error': 'Initial synthesis failed', 'orchestration_log': orchestration_log}
                
                # Stage 4: Critique and Refinement
                if enable_critique:
                    logger.info("Stage 4: Critique and refinement...")
                    
                    max_critique_rounds = self.deliberation_config.get('max_critique_rounds', 2)
                    current_response = synthesis_result['response']
                    
                    for round_num in range(1, max_critique_rounds + 1):
                        logger.info(f"Critique round {round_num}/{max_critique_rounds}")
                        
                        # Critique phase
                        critique_prompt = f"""
Original Query: {query}

Current Response Draft:
{current_response}

Planning Context:
{planning_result['response']}

Research Context:
{research_result['response']}

As the Quality Assurance Critic, your task is to:
1. Thoroughly review the current response for accuracy, completeness, and quality
2. Identify any errors, biases, or missing information
3. Check if the response fully addresses the original query
4. Evaluate the logical flow and clarity
5. Suggest specific, actionable improvements

Provide constructive criticism and specific recommendations for improvement. Be thorough but fair in your analysis.
"""
                        
                        critique_result = await self._execute_agent_stage(critic, critique_prompt, session, f"Critique Round {round_num}")
                        orchestration_log['stages'].append(critique_result)
                        orchestration_log['critique_rounds'] += 1
                        
                        if not critique_result['success']:
                            logger.warning(f"Critique round {round_num} failed, proceeding with current response")
                            break
                        
                        # Refinement phase
                        refinement_prompt = f"""
Original Query: {query}

Current Response:
{current_response}

Critique and Suggestions:
{critique_result['response']}

As the Content Synthesizer, your task is to:
1. Carefully review the critic's feedback
2. Revise and improve your response based on the constructive criticism
3. Address any identified errors or gaps
4. Enhance clarity and completeness
5. Ensure the response fully addresses the original query

Provide a refined, improved version of your response that incorporates the critic's valuable feedback.
"""
                        
                        refinement_result = await self._execute_agent_stage(synthesizer, refinement_prompt, session, f"Refinement Round {round_num}")
                        orchestration_log['stages'].append(refinement_result)
                        
                        if refinement_result['success']:
                            current_response = refinement_result['response']
                        else:
                            logger.warning(f"Refinement round {round_num} failed, keeping previous response")
                            break
                    
                    final_response = current_response
                else:
                    final_response = synthesis_result['response']
                
                # Calculate total usage
                total_usage = self._calculate_orchestration_usage(orchestration_log)
                
                return {
                    'success': True,
                    'final_response': final_response,
                    'orchestration_log': orchestration_log,
                    'total_usage': total_usage
                }
                
            except Exception as e:
                logger.error(f"Sequential orchestration error: {str(e)}")
                return {
                    'success': False,
                    'error': str(e),
                    'orchestration_log': orchestration_log
                }

    async def _orchestrate_parallel(self, query: str, session_id: str, force_agent_count: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced parallel orchestration with specialized roles and critique rounds."""
        logger.info("Using enhanced parallel orchestration with specialized roles...")
        
        # Find specialized agents - prioritize researcher and synthesizer for parallel execution
        researcher = self._find_agent_by_role('researcher')
        critic = self._find_agent_by_role('critic')
        synthesizer = self._find_agent_by_role('synthesizer')
        
        # Fallback to all agents if specialized ones not found
        if not all([researcher, synthesizer]):
            logger.warning("Specialized agents not found, using all available agents")
            active_workers = self.workers[:3]  # Use first 3 agents
        else:
            # Use researcher + synthesizer for parallel execution
            active_workers = [researcher, synthesizer]
            logger.info(f"Using specialized agents: {[w.name for w in active_workers]}")
        
        # Execute workers in parallel
        worker_results = []
        timeout = self.system_config.get('task_timeout', 120)
        
        async with aiohttp.ClientSession() as session:
            # Phase 1: Parallel execution of researcher and synthesizer
            tasks = []
            for worker in active_workers:
                # Customize prompt based on role
                if hasattr(worker, 'agent_config') and worker.agent_config.get('role') == 'researcher':
                    worker_prompt = f"""
{query}

As the Research Specialist, focus on gathering comprehensive, accurate information using available tools when needed. 
Provide well-researched facts and data that directly address this query.
"""
                else:
                    worker_prompt = query
                
                task = asyncio.create_task(
                    self._execute_worker_with_timeout(worker, worker_prompt, session, timeout)
                )
                tasks.append(task)
            
            # Wait for parallel execution
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    worker_results.append({
                        'success': False,
                        'agent_name': active_workers[i].name,
                        'model': active_workers[i].model,
                        'response': None,
                        'tool_usage': [],
                        'tool_costs': {},
                        'processing_time': 0,
                        'error': str(result)
                    })
                else:
                    worker_results.append(result)
            
            # Filter successful results
            successful_results = [r for r in worker_results if r['success']]
            
            if not successful_results:
                return {
                    'success': False,
                    'error': 'All workers failed',
                    'orchestration_log': {
                        'mode': 'parallel_enhanced',
                        'worker_results': worker_results,
                        'stages': []
                    }
                }
            
            # Phase 2: Fast critique and refinement (if enabled and critic available)
            enable_critique = self.deliberation_config.get('enable_critique_rounds', True)
            final_response_content = None
            
            if enable_critique and critic and len(successful_results) > 0:
                logger.info("Performing fast critique and refinement...")
                
                # Combine the parallel responses
                combined_responses = []
                for result in successful_results:
                    agent_name = result['agent_name']
                    response = result['response']
                    combined_responses.append(f"Agent {agent_name}: {response}")
                
                # Quick critique
                critique_prompt = f"""
Original Query: {query}

Agent Responses:
{chr(10).join(combined_responses)}

As the Quality Critic, quickly identify the key strengths and any critical issues with these responses. 
Focus on: accuracy, completeness, and relevance to the query.
Be concise - provide 2-3 key points for improvement or validation.
"""
                
                critique_result = await self._execute_worker_with_timeout(critic, critique_prompt, session, timeout)
                
                if critique_result['success']:
                    # Quick synthesis with critique
                    synthesis_prompt = f"""
Original Query: {query}

Agent Responses:
{chr(10).join(combined_responses)}

Critic's Feedback:
{critique_result['response']}

Create a final, high-quality response that incorporates the best information from the agents and addresses the critic's feedback.
Be comprehensive but concise.
"""
                    
                    synthesis_result = await self._execute_worker_with_timeout(synthesizer, synthesis_prompt, session, timeout)
                    
                    if synthesis_result['success']:
                        final_response_content = synthesis_result['response']
                        worker_results.extend([critique_result, synthesis_result])
                    else:
                        logger.warning("Synthesis failed, using best individual response")
                        final_response_content = max(successful_results, key=lambda x: len(x.get('response', '')))['response']
                else:
                    logger.warning("Critique failed, using direct synthesis")
                    final_response_content = await self._quick_synthesis(successful_results, query)
            else:
                # Direct synthesis without critique
                final_response_content = await self._quick_synthesis(successful_results, query)
            
            # Calculate total usage
            total_usage = self._calculate_parallel_usage(worker_results)
            
            return {
                'success': True,
                'final_response': final_response_content,
                'orchestration_log': {
                    'mode': 'parallel_enhanced',
                    'worker_results': worker_results,
                    'agents_used': [w.name for w in active_workers],
                    'critique_enabled': enable_critique and critic is not None
                },
                'total_usage': total_usage
            }

    async def _quick_synthesis(self, successful_results: List[Dict[str, Any]], query: str) -> str:
        """Quick synthesis of multiple agent responses."""
        if len(successful_results) == 1:
            return successful_results[0]['response']
        
        # Combine responses intelligently
        combined_responses = []
        for result in successful_results:
            agent_name = result['agent_name']
            response = result['response']
            combined_responses.append(f"Agent {agent_name}: {response}")
        
        # Simple synthesis logic - take the longest response as primary and enhance with others
        primary_response = max(successful_results, key=lambda x: len(x.get('response', '')))['response']
        
        return primary_response

    def _find_agent_by_role(self, role: str) -> Optional[WorkerAgent]:
        """Find an agent by their specialized role."""
        for worker in self.workers:
            if worker.agent_config.get('role') == role:
                return worker
        return None

    async def _execute_agent_stage(self, agent: WorkerAgent, prompt: str, session: aiohttp.ClientSession, stage_name: str) -> Dict[str, Any]:
        """Execute a specific agent for a stage in the orchestration."""
        logger.info(f"Executing {stage_name} with {agent.name}...")
        
        timeout = self.system_config.get('task_timeout', 60)
        stage_start = asyncio.get_event_loop().time()
        
        try:
            result = await asyncio.wait_for(
                agent.process_task(prompt, session),
                timeout=timeout
            )
            
            stage_end = asyncio.get_event_loop().time()
            
            return {
                'stage_name': stage_name,
                'agent_name': agent.name,
                'agent_role': agent.agent_config.get('role', 'unknown'),
                'success': result['success'],
                'response': result.get('response', ''),
                'tool_usage': result.get('tool_usage', []),
                'tool_costs': result.get('tool_costs', {}),
                'usage_stats': result.get('usage_stats', {}),
                'processing_time': stage_end - stage_start,
                'error': result.get('error')
            }
            
        except asyncio.TimeoutError:
            return {
                'stage_name': stage_name,
                'agent_name': agent.name,
                'agent_role': agent.agent_config.get('role', 'unknown'),
                'success': False,
                'response': '',
                'tool_usage': [],
                'tool_costs': {},
                'usage_stats': {},
                'processing_time': timeout,
                'error': f'{stage_name} timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'stage_name': stage_name,
                'agent_name': agent.name,
                'agent_role': agent.agent_config.get('role', 'unknown'),
                'success': False,
                'response': '',
                'tool_usage': [],
                'tool_costs': {},
                'usage_stats': {},
                'processing_time': 0,
                'error': str(e)
            }
    
    async def _execute_workers(self, query: str, session_id: str, agent_count: int = None) -> List[Dict[str, Any]]:
        """Execute worker agents in parallel with dynamic scaling."""
        if agent_count is None:
            agent_count = len(self.workers)
        
        # Select subset of workers based on agent count
        active_workers = self.workers[:agent_count]
        logger.info(f"Executing {len(active_workers)} of {len(self.workers)} workers in parallel...")
        
        timeout = self.system_config.get('task_timeout', 60)
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for parallel execution
            tasks = []
            for worker in active_workers:
                task = asyncio.create_task(
                    self._execute_worker_with_timeout(worker, query, session, timeout)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            worker_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    worker_results.append({
                        'success': False,
                        'agent_name': active_workers[i].name,
                        'model': active_workers[i].model,
                        'response': None,
                        'tool_usage': [],
                        'processing_time': 0,
                        'error': str(result)
                    })
                else:
                    worker_results.append(result)
            
            return worker_results
    
    async def _execute_worker_with_timeout(self, worker: WorkerAgent, query: str, 
                                          session: aiohttp.ClientSession, timeout: int) -> Dict[str, Any]:
        """Execute a worker with timeout handling."""
        try:
            return await asyncio.wait_for(
                worker.process_task(query, session), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Worker {worker.name} timed out after {timeout} seconds")
            return {
                'success': False,
                'agent_name': worker.name,
                'model': worker.model,
                'response': None,
                'tool_usage': [],
                'processing_time': timeout,
                'error': f'Task timed out after {timeout} seconds'
            }
    
    async def _deliberate(self, worker_results: List[Dict[str, Any]], original_query: str) -> Dict[str, Any]:
        """Deliberate on worker results and synthesize final response."""
        logger.info("Starting deliberation process...")
        
        # Prepare responses for synthesis
        responses_text = []
        for result in worker_results:
            agent_name = result['agent_name']
            response = result['response']
            responses_text.append(f"Agent {agent_name}: {response}")
        
        # Create synthesis prompt
        synthesis_prompt = f"""
You are a synthesis agent responsible for creating the best possible response from multiple AI agent outputs.

Original Query: {original_query}

Agent Responses:
{chr(10).join(responses_text)}

Your task:
1. Analyze all the agent responses carefully
2. Identify the most accurate, comprehensive, and helpful information
3. Synthesize a single, high-quality response that combines the best elements
4. Ensure the final response is coherent, well-structured, and directly addresses the user's query
5. If there are conflicting information, use your judgment to determine the most reliable sources

Provide only the final synthesized response. Do not include meta-commentary about the synthesis process.
"""
        
        # Make synthesis API call
        synthesis_model = self.deliberation_config.get('synthesis_model', 'openai/gpt-4o')
        synthesis_result = await self._call_synthesis_llm(synthesis_prompt, synthesis_model)
        
        if synthesis_result['success']:
            return {
                'content': synthesis_result['result'],
                'deliberation_log': {
                    'synthesis_model': synthesis_model,
                    'worker_count': len(worker_results),
                    'synthesis_prompt': synthesis_prompt,
                    'synthesis_success': True,
                    'synthesis_usage': synthesis_result.get('usage_stats', {})
                }
            }
        else:
            # Fallback: return the best individual response
            logger.warning("Synthesis failed, falling back to best individual response")
            best_result = max(worker_results, key=lambda x: len(x.get('response', '')))
            return {
                'content': best_result['response'],
                'deliberation_log': {
                    'synthesis_model': synthesis_model,
                    'worker_count': len(worker_results),
                    'synthesis_success': False,
                    'synthesis_error': synthesis_result['error'],
                    'fallback_agent': best_result['agent_name']
                }
            }
    
    async def _call_synthesis_llm(self, prompt: str, model: str) -> Dict[str, Any]:
        """Make an API call for synthesis."""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a synthesis expert that combines multiple AI responses into the best possible single response.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,  # Lower temperature for more focused synthesis
            'max_tokens': 4000,  # Increased for Grok reasoning
            'usage': {
                'include': True  # Enable usage tracking
            }
        }
        
        api_config = self.config.get('api', {})
        base_url = api_config.get('base_url', 'https://openrouter.ai/api/v1')
        timeout = api_config.get('timeout', 60)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        usage_stats = data.get('usage', {})
                        
                        # Log synthesis usage
                        if usage_stats:
                            logger.info(f"Synthesis: Usage - Input: {usage_stats.get('prompt_tokens', 0)} tokens, "
                                      f"Output: {usage_stats.get('completion_tokens', 0)} tokens, "
                                      f"Total: {usage_stats.get('total_tokens', 0)} tokens")
                        
                        return {
                            'success': True,
                            'result': content,
                            'error': None,
                            'usage_stats': usage_stats
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'result': None,
                            'error': f"Synthesis API call failed: {response.status} - {error_text}"
                        }
                        
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'error': str(e)
            }
    
    async def _save_session_log(self, result: Dict[str, Any]) -> None:
        """Save session log to file."""
        try:
            session_id = result['session_id']
            log_file = os.path.join(self.logs_dir, f"session_{session_id}.json")
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                
            logger.info(f"Session log saved: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session log: {str(e)}")
    
    def _display_results(self, result: Dict[str, Any], verbose: bool) -> None:
        """Display results to the user."""
        print("\n" + "="*80)
        print("GROK HEAVY - MULTI-AGENT AI SYSTEM RESPONSE")
        print("="*80)
        
        if result['success']:
            print(f"\n{result['final_response']}")
            
            # Show cost summary even in non-verbose mode
            if 'total_usage' in result and 'costs' in result['total_usage']:
                costs = result['total_usage']['costs']
                tool_usage = result['total_usage'].get('tool_usage', {})
                total_cost = costs.get('total_cost', 0)
                agents_used = result.get('agents_used', 0)
                
                # Build cost summary
                cost_parts = []
                if tool_usage.get('exa_searches', 0) > 0:
                    cost_parts.append(f"{tool_usage['exa_searches']} searches")
                if tool_usage.get('code_executions', 0) > 0:
                    cost_parts.append(f"{tool_usage['code_executions']} executions")
                if tool_usage.get('data_analyses', 0) > 0:
                    cost_parts.append(f"{tool_usage['data_analyses']} analyses")
                
                tools_info = f", {', '.join(cost_parts)}" if cost_parts else ""
                print(f"\nQuery Cost: ${total_cost:.6f} ({agents_used} agent{'s' if agents_used != 1 else ''}, {result['total_usage'].get('total_tokens', 0):,} tokens{tools_info})")
            
            if verbose:
                print("\n" + "-"*60)
                print("DETAILED DELIBERATION LOG")
                print("-"*60)
                
                print(f"\nSession ID: {result['session_id']}")
                print(f"Processing Time: {result['processing_time']:.2f} seconds")
                print(f"Workers Used: {len([r for r in result['worker_results'] if r['success']])}/{len(result['worker_results'])}")
                
                # Display usage statistics
                if 'total_usage' in result:
                    usage = result['total_usage']
                    print(f"\nToken Usage Summary:")
                    print(f"   Input Tokens: {usage.get('prompt_tokens', 0):,}")
                    print(f"   Output Tokens: {usage.get('completion_tokens', 0):,}")
                    print(f"   Total Tokens: {usage.get('total_tokens', 0):,}")
                    
                    # Display real costs
                    if 'costs' in usage:
                        costs = usage['costs']
                        tool_usage = usage.get('tool_usage', {})
                        print(f"\nCost Breakdown:")
                        print(f"   LLM Input Cost: ${costs.get('llm_input_cost', 0):.6f}")
                        print(f"   LLM Output Cost: ${costs.get('llm_output_cost', 0):.6f}")
                        print(f"   LLM Total: ${costs.get('llm_total_cost', 0):.6f}")
                        
                        if tool_usage.get('total_tool_cost', 0) > 0:
                            print(f"   Tool Costs: ${tool_usage['total_tool_cost']:.6f}")
                            if tool_usage.get('exa_searches', 0) > 0:
                                print(f"     Exa Searches: {tool_usage['exa_searches']} × $0.005 = ${tool_usage['exa_searches'] * 0.005:.6f}")
                            if tool_usage.get('code_executions', 0) > 0:
                                print(f"     Code Executions: {tool_usage['code_executions']} (free)")
                            if tool_usage.get('data_analyses', 0) > 0:
                                print(f"     Data Analyses: {tool_usage['data_analyses']} (free)")
                        
                        print(f"   Total Cost: ${costs.get('total_cost', 0):.6f}")
                        
                        # Show cost per agent
                        if 'breakdown' in usage:
                            breakdown = usage['breakdown']
                            if breakdown.get('workers'):
                                print(f"\n   Per-Agent Costs:")
                                for worker in breakdown['workers']:
                                    agent_name = worker.get('agent', 'unknown')
                                    agent_total_cost = worker.get('total_cost', 0)
                                    agent_llm_cost = worker.get('llm_cost', {}).get('total_cost', 0)
                                    agent_tool_costs = worker.get('tool_costs', {})
                                    agent_tool_cost = agent_tool_costs.get('costs', {}).get('total_tool_cost', 0)
                                    
                                    cost_breakdown = f"${agent_total_cost:.6f}"
                                    if agent_tool_cost > 0:
                                        cost_breakdown = f"${agent_total_cost:.6f} (LLM: ${agent_llm_cost:.6f}, Tools: ${agent_tool_cost:.6f})"
                                    print(f"     {agent_name}: {cost_breakdown}")
                                
                                synthesis_cost = breakdown.get('synthesis', {}).get('cost', {}).get('total_cost', 0)
                                if synthesis_cost > 0:
                                    print(f"     synthesis: ${synthesis_cost:.6f}")
                    
                    # Show complexity analysis
                    if 'complexity_analysis' in result:
                        complexity = result['complexity_analysis']
                        print(f"\nComplexity Analysis:")
                        print(f"   Level: {complexity.get('complexity_level', 'unknown')}")
                        print(f"   Score: {complexity.get('complexity_score', 0)}")
                        print(f"   Agents Used: {result.get('agents_used', 0)}/3")
                
                print("\nIndividual Agent Responses:")
                for worker_result in result['worker_results']:
                    status = "✓" if worker_result['success'] else "✗"
                    print(f"\n{status} {worker_result['agent_name']} ({worker_result['model']}):")
                    if worker_result['success']:
                        print(f"   Response: {worker_result['response'][:200]}...")
                        print(f"   Processing Time: {worker_result['processing_time']:.2f}s")
                        if worker_result['tool_usage']:
                            print(f"   Tools Used: {len(worker_result['tool_usage'])} tool calls")
                        if 'usage_stats' in worker_result:
                            usage = worker_result['usage_stats']
                            print(f"   Tokens: {usage.get('total_tokens', 0)} total ({usage.get('prompt_tokens', 0)} in + {usage.get('completion_tokens', 0)} out)")
                    else:
                        print(f"   Error: {worker_result['error']}")
                
                if result['deliberation_log']:
                    print("\nDeliberation Summary:")
                    log = result['deliberation_log']
                    print(f"   Synthesis Model: {log.get('synthesis_model')}")
                    print(f"   Synthesis Success: {log.get('synthesis_success')}")
                    if not log.get('synthesis_success'):
                        print(f"   Fallback Agent: {log.get('fallback_agent')}")
        else:
            print(f"\nERROR: {result['error']}")
        
        print("\n" + "="*80) 

    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity and determine dynamic spawning needs."""
        complexity_score = 0
        complexity_factors = []
        spawning_recommendations = []
        
        # Get dynamic spawning config
        spawning_config = self.config.get('dynamic_spawning', {})
        if not spawning_config.get('enabled', False):
            # Fallback to original logic if spawning disabled
            return self._legacy_complexity_analysis(query)
        
        triggers = spawning_config.get('spawning_triggers', {})
        query_lower = query.lower()
        
        # Length-based complexity
        if len(query) > 200:
            complexity_score += 2
            complexity_factors.append("long_query")
        elif len(query) > 100:
            complexity_score += 1
            complexity_factors.append("medium_query")
        
        # Keyword-based complexity indicators
        complex_keywords = [
            'analyze', 'compare', 'research', 'explain', 'detailed', 'comprehensive',
            'search for', 'find information', 'latest', 'recent', 'current',
            'pros and cons', 'advantages', 'disadvantages', 'benefits', 'risks'
        ]
        
        simple_keywords = [
            'what is', 'define', 'meaning', 'simple', 'basic', 'quick',
            'yes or no', 'true or false', 'which', 'when', 'where'
        ]
        
        complex_matches = sum(1 for keyword in complex_keywords if keyword in query_lower)
        simple_matches = sum(1 for keyword in simple_keywords if keyword in query_lower)
        
        complexity_score += complex_matches * 2
        complexity_score -= simple_matches
        
        if complex_matches > 0:
            complexity_factors.extend([f"complex_keyword_{i}" for i in range(complex_matches)])
        if simple_matches > 0:
            complexity_factors.extend([f"simple_keyword_{i}" for i in range(simple_matches)])
        
        # Dynamic spawning triggers
        
        # 1. Comparison detection
        comparison_keywords = triggers.get('comparison_keywords', [])
        comparison_detected = any(keyword in query_lower for keyword in comparison_keywords)
        if comparison_detected:
            complexity_score += 3
            complexity_factors.append("comparison_task")
            
            # Detect what's being compared
            comparison_subjects = self._extract_comparison_subjects(query)
            for subject in comparison_subjects:
                spawning_recommendations.append({
                    'template': 'comparative_researcher',
                    'focus_area': subject,
                    'reason': f'Comparative analysis of {subject}'
                })
        
        # 2. Multi-task detection
        multi_task_keywords = triggers.get('multi_task_keywords', [])
        multi_task_detected = any(keyword in query_lower for keyword in multi_task_keywords)
        if multi_task_detected:
            complexity_score += 2
            complexity_factors.append("multi_task")
            
            # Detect coding requirements
            coding_keywords = triggers.get('coding_keywords', [])
            coding_detected = any(keyword in query_lower for keyword in coding_keywords)
            if coding_detected:
                complexity_score += 2
                complexity_factors.append("coding_required")
                spawning_recommendations.append({
                    'template': 'coder_specialist',
                    'focus_area': 'programming',
                    'reason': 'Code generation and visualization required'
                })
        
        # 3. Geographic/regional analysis
        geographic_keywords = triggers.get('geographic_keywords', [])
        geographic_matches = [kw for kw in geographic_keywords if kw in query_lower]
        if len(geographic_matches) >= 2:  # Multiple regions mentioned
            complexity_score += 2
            complexity_factors.append("multi_regional")
            
            for region in geographic_matches:
                spawning_recommendations.append({
                    'template': 'geographic_researcher',
                    'focus_area': region,
                    'reason': f'Regional analysis for {region}'
                })
        
        # Question complexity
        question_marks = query.count('?')
        if question_marks > 2:
            complexity_score += 2
            complexity_factors.append("multiple_questions")
        elif question_marks == 0:
            complexity_score += 1
            complexity_factors.append("statement_query")
        
        # Tool usage indicators
        tool_indicators = ['search', 'calculate', 'code', 'execute', 'run', 'compute']
        if any(indicator in query_lower for indicator in tool_indicators):
            complexity_score += 2
            complexity_factors.append("tool_usage_expected")
        
        # Determine if spawning is needed
        threshold = triggers.get('complexity_threshold', 6)
        should_spawn = complexity_score >= threshold and len(spawning_recommendations) > 0
        
        # Determine agent count based on complexity and spawning
        if should_spawn:
            max_spawned = spawning_config.get('max_spawned_agents', 6)
            recommended_agents = min(len(self.workers) + len(spawning_recommendations), max_spawned)
            complexity_level = "complex_with_spawning"
        else:
            # Original logic for non-spawning scenarios
            if complexity_score <= 1:
                recommended_agents = 1
                complexity_level = "simple"
            elif complexity_score <= 4:
                recommended_agents = 2
                complexity_level = "medium"
            else:
                recommended_agents = 3
                complexity_level = "complex"
        
        return {
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'recommended_agents': recommended_agents,
            'factors': complexity_factors,
            'query_length': len(query),
            'should_spawn_agents': should_spawn,
            'spawning_recommendations': spawning_recommendations,
            'spawning_enabled': spawning_config.get('enabled', False)
        }

    def _extract_comparison_subjects(self, query: str) -> List[str]:
        """Extract subjects being compared from a query."""
        subjects = []
        query_lower = query.lower()
        
        # Common comparison patterns
        patterns = [
            r'compare\s+(.+?)\s+(?:vs|versus|and|with)\s+(.+?)(?:\s|$|\.|\?)',
            r'(.+?)\s+(?:vs|versus)\s+(.+?)(?:\s|$|\.|\?)',
            r'difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\s|$|\.|\?)',
            r'contrast\s+(.+?)\s+(?:with|and)\s+(.+?)(?:\s|$|\.|\?)'
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                subjects.extend([s.strip() for s in match if s.strip()])
        
        # Clean up subjects (remove common words)
        cleaned_subjects = []
        stop_words = {'the', 'of', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with'}
        for subject in subjects:
            words = subject.split()
            cleaned = ' '.join([w for w in words if w not in stop_words])
            if cleaned and len(cleaned) > 2:
                cleaned_subjects.append(cleaned)
        
        return list(set(cleaned_subjects))  # Remove duplicates

    def _legacy_complexity_analysis(self, query: str) -> Dict[str, Any]:
        """Original complexity analysis for when dynamic spawning is disabled."""
        complexity_score = 0
        complexity_factors = []
        
        # Length-based complexity
        if len(query) > 200:
            complexity_score += 2
            complexity_factors.append("long_query")
        elif len(query) > 100:
            complexity_score += 1
            complexity_factors.append("medium_query")
        
        # Keyword-based complexity indicators
        complex_keywords = [
            'analyze', 'compare', 'research', 'explain', 'detailed', 'comprehensive',
            'search for', 'find information', 'latest', 'recent', 'current',
            'pros and cons', 'advantages', 'disadvantages', 'benefits', 'risks'
        ]
        
        simple_keywords = [
            'what is', 'define', 'meaning', 'simple', 'basic', 'quick',
            'yes or no', 'true or false', 'which', 'when', 'where'
        ]
        
        query_lower = query.lower()
        
        complex_matches = sum(1 for keyword in complex_keywords if keyword in query_lower)
        simple_matches = sum(1 for keyword in simple_keywords if keyword in query_lower)
        
        complexity_score += complex_matches * 2
        complexity_score -= simple_matches
        
        if complex_matches > 0:
            complexity_factors.extend([f"complex_keyword_{i}" for i in range(complex_matches)])
        if simple_matches > 0:
            complexity_factors.extend([f"simple_keyword_{i}" for i in range(simple_matches)])
        
        # Question complexity
        question_marks = query.count('?')
        if question_marks > 2:
            complexity_score += 2
            complexity_factors.append("multiple_questions")
        elif question_marks == 0:
            complexity_score += 1
            complexity_factors.append("statement_query")
        
        # Tool usage indicators
        tool_indicators = ['search', 'calculate', 'code', 'execute', 'run', 'compute']
        if any(indicator in query_lower for indicator in tool_indicators):
            complexity_score += 2
            complexity_factors.append("tool_usage_expected")
        
        # Determine agent count based on complexity
        if complexity_score <= 1:
            agent_count = 1
            complexity_level = "simple"
        elif complexity_score <= 4:
            agent_count = 2
            complexity_level = "medium"
        else:
            agent_count = 3
            complexity_level = "complex"
        
        return {
            'complexity_score': complexity_score,
            'complexity_level': complexity_level,
            'recommended_agents': agent_count,
            'factors': complexity_factors,
            'query_length': len(query),
            'should_spawn_agents': False,
            'spawning_recommendations': [],
            'spawning_enabled': False
        }
    
    def _calculate_total_usage(self, worker_results: List[Dict[str, Any]], deliberation_log: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total token usage across all agents and synthesis with real cost including tool costs."""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        # Track tool costs
        total_tool_cost = 0.0
        total_exa_searches = 0
        total_code_executions = 0
        total_data_analyses = 0
        
        # Sum up worker usage
        for result in worker_results:
            if result.get('success') and 'usage_stats' in result:
                usage = result['usage_stats']
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)
            
            # Sum up tool costs
            if result.get('success') and 'tool_costs' in result:
                tool_costs = result['tool_costs']
                total_tool_cost += tool_costs.get('costs', {}).get('total_tool_cost', 0.0)
                total_exa_searches += tool_costs.get('exa_searches', 0)
                total_code_executions += tool_costs.get('code_executions', 0)
                total_data_analyses += tool_costs.get('data_analyses', 0)
        
        # Add synthesis usage
        synthesis_usage = deliberation_log.get('synthesis_usage', {})
        if synthesis_usage:
            total_prompt_tokens += synthesis_usage.get('prompt_tokens', 0)
            total_completion_tokens += synthesis_usage.get('completion_tokens', 0)
            total_tokens += synthesis_usage.get('total_tokens', 0)
        
        # Calculate real costs using Grok 3 mini beta pricing
        # $0.30/Million input tokens, $0.50/Million output tokens
        llm_input_cost = (total_prompt_tokens / 1_000_000) * 0.30
        llm_output_cost = (total_completion_tokens / 1_000_000) * 0.50
        llm_total_cost = llm_input_cost + llm_output_cost
        
        # Total cost includes LLM + tools
        total_cost = llm_total_cost + total_tool_cost
        
        return {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens,
            'costs': {
                'llm_input_cost': llm_input_cost,
                'llm_output_cost': llm_output_cost,
                'llm_total_cost': llm_total_cost,
                'tool_cost': total_tool_cost,
                'total_cost': total_cost,
                'currency': 'USD',
                'pricing_model': 'grok-3-mini-beta + tools'
            },
            'tool_usage': {
                'exa_searches': total_exa_searches,
                'code_executions': total_code_executions,
                'data_analyses': total_data_analyses,
                'total_tool_cost': total_tool_cost
            },
            'agents_used': len([r for r in worker_results if r.get('success')]),
            'breakdown': {
                'workers': [
                    {
                        'agent': result.get('agent_name', 'unknown'),
                        'usage': result.get('usage_stats', {}),
                        'llm_cost': self._calculate_individual_cost(result.get('usage_stats', {})),
                        'tool_costs': result.get('tool_costs', {}),
                        'total_cost': (
                            self._calculate_individual_cost(result.get('usage_stats', {})).get('total_cost', 0.0) +
                            result.get('tool_costs', {}).get('costs', {}).get('total_tool_cost', 0.0)
                        )
                    }
                    for result in worker_results if result.get('success')
                ],
                'synthesis': {
                    **synthesis_usage,
                    'cost': self._calculate_individual_cost(synthesis_usage)
                }
            }
        }
    
    def _calculate_individual_cost(self, usage_stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cost for individual agent usage."""
        if not usage_stats:
            return {'input_cost': 0.0, 'output_cost': 0.0, 'total_cost': 0.0}
        
        prompt_tokens = usage_stats.get('prompt_tokens', 0)
        completion_tokens = usage_stats.get('completion_tokens', 0)
        
        input_cost = (prompt_tokens / 1_000_000) * 0.30
        output_cost = (completion_tokens / 1_000_000) * 0.50
        total_cost = input_cost + output_cost
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        } 

    def _display_results_enhanced(self, result: Dict[str, Any], verbose: bool) -> None:
        """Enhanced display for the new orchestration system."""
        print("\n" + "="*80)
        print("GROK HEAVY - MULTI-AGENT AI SYSTEM RESPONSE")
        print("="*80)
        
        if result['success']:
            print(f"\n{result['final_response']}")
            
            # Show cost summary
            if 'total_usage' in result and 'costs' in result['total_usage']:
                costs = result['total_usage']['costs']
                tool_usage = result['total_usage'].get('tool_usage', {})
                total_cost = costs.get('total_cost', 0)
                agents_used = result['total_usage'].get('agents_used', 0)
                stages_completed = result['total_usage'].get('stages_completed', 0)
                
                # Build cost summary
                cost_parts = []
                if tool_usage.get('exa_searches', 0) > 0:
                    cost_parts.append(f"{tool_usage['exa_searches']} searches")
                if tool_usage.get('code_executions', 0) > 0:
                    cost_parts.append(f"{tool_usage['code_executions']} executions")
                if tool_usage.get('data_analyses', 0) > 0:
                    cost_parts.append(f"{tool_usage['data_analyses']} analyses")
                
                tools_info = f", {', '.join(cost_parts)}" if cost_parts else ""
                orchestration_mode = result.get('orchestration_mode', 'unknown')
                print(f"\nQuery Cost: ${total_cost:.6f} ({orchestration_mode} mode, {stages_completed} stages, {result['total_usage'].get('total_tokens', 0):,} tokens{tools_info})")
            
            if verbose:
                print("\n" + "-"*60)
                print("DETAILED ORCHESTRATION LOG")
                print("-"*60)
                
                orchestration_log = result.get('orchestration_log', {})
                mode = orchestration_log.get('mode', 'unknown')
                
                print(f"\nSession ID: {result['session_id']}")
                print(f"Processing Time: {result['processing_time']:.2f} seconds")
                print(f"Orchestration Mode: {mode}")
                
                if mode == 'sequential':
                    self._display_sequential_details(result, orchestration_log)
                else:
                    self._display_parallel_details(result, orchestration_log)
        else:
            print(f"\nERROR: {result['error']}")
        
        print("\n" + "="*80)

    def _display_sequential_details(self, result: Dict[str, Any], orchestration_log: Dict[str, Any]) -> None:
        """Display detailed information for sequential orchestration."""
        agents_used = orchestration_log.get('agents_used', [])
        critique_rounds = orchestration_log.get('critique_rounds', 0)
        stages = orchestration_log.get('stages', [])
        
        print(f"Specialized Agents: {', '.join(agents_used)}")
        print(f"Critique Rounds: {critique_rounds}")
        print(f"Total Stages: {len(stages)}")
        
        # Display usage statistics
        if 'total_usage' in result:
            usage = result['total_usage']
            print(f"\nToken Usage Summary:")
            print(f"   Input Tokens: {usage.get('prompt_tokens', 0):,}")
            print(f"   Output Tokens: {usage.get('completion_tokens', 0):,}")
            print(f"   Total Tokens: {usage.get('total_tokens', 0):,}")
            
            # Display cost breakdown
            if 'costs' in usage:
                costs = usage['costs']
                tool_usage = usage.get('tool_usage', {})
                print(f"\nCost Breakdown:")
                print(f"   LLM Input Cost: ${costs.get('llm_input_cost', 0):.6f}")
                print(f"   LLM Output Cost: ${costs.get('llm_output_cost', 0):.6f}")
                print(f"   LLM Total: ${costs.get('llm_total_cost', 0):.6f}")
                
                if tool_usage.get('total_tool_cost', 0) > 0:
                    print(f"   Tool Costs: ${tool_usage['total_tool_cost']:.6f}")
                    if tool_usage.get('exa_searches', 0) > 0:
                        print(f"     Exa Searches: {tool_usage['exa_searches']} × $0.005 = ${tool_usage['exa_searches'] * 0.005:.6f}")
                    if tool_usage.get('code_executions', 0) > 0:
                        print(f"     Code Executions: {tool_usage['code_executions']} (free)")
                    if tool_usage.get('data_analyses', 0) > 0:
                        print(f"     Data Analyses: {tool_usage['data_analyses']} (free)")
                
                print(f"   Total Cost: ${costs.get('total_cost', 0):.6f}")
        
        # Display stage-by-stage breakdown
        print("\nStage-by-Stage Execution:")
        for i, stage in enumerate(stages, 1):
            status = "✓" if stage['success'] else "✗"
            stage_name = stage.get('stage_name', 'Unknown')
            agent_name = stage.get('agent_name', 'Unknown')
            agent_role = stage.get('agent_role', 'Unknown')
            processing_time = stage.get('processing_time', 0)
            
            print(f"\n{status} Stage {i}: {stage_name}")
            print(f"   Agent: {agent_name} ({agent_role})")
            print(f"   Processing Time: {processing_time:.2f}s")
            
            if stage['success']:
                if stage.get('tool_usage'):
                    print(f"   Tools Used: {len(stage['tool_usage'])} tool calls")
                if 'usage_stats' in stage:
                    usage = stage['usage_stats']
                    print(f"   Tokens: {usage.get('total_tokens', 0)} total ({usage.get('prompt_tokens', 0)} in + {usage.get('completion_tokens', 0)} out)")
            else:
                print(f"   Error: {stage.get('error', 'Unknown error')}")

    def _display_parallel_details(self, result: Dict[str, Any], orchestration_log: Dict[str, Any]) -> None:
        """Display detailed information for parallel orchestration (legacy mode)."""
        worker_results = orchestration_log.get('worker_results', [])
        complexity_analysis = orchestration_log.get('complexity_analysis', {})
        
        print(f"Workers Used: {len([r for r in worker_results if r['success']])}/{len(worker_results)}")
        
        # Show complexity analysis
        if complexity_analysis:
            print(f"\nComplexity Analysis:")
            print(f"   Level: {complexity_analysis.get('complexity_level', 'unknown')}")
            print(f"   Score: {complexity_analysis.get('complexity_score', 0)}")
        
        # Display usage statistics
        if 'total_usage' in result:
            usage = result['total_usage']
            print(f"\nToken Usage Summary:")
            print(f"   Input Tokens: {usage.get('prompt_tokens', 0):,}")
            print(f"   Output Tokens: {usage.get('completion_tokens', 0):,}")
            print(f"   Total Tokens: {usage.get('total_tokens', 0):,}")
        
        # Display individual worker results
        print("\nIndividual Agent Responses:")
        for worker_result in worker_results:
            status = "✓" if worker_result['success'] else "✗"
            print(f"\n{status} {worker_result['agent_name']} ({worker_result['model']}):")
            if worker_result['success']:
                print(f"   Response: {worker_result['response'][:200]}...")
                print(f"   Processing Time: {worker_result['processing_time']:.2f}s")
                if worker_result.get('tool_usage'):
                    print(f"   Tools Used: {len(worker_result['tool_usage'])} tool calls")
            else:
                print(f"   Error: {worker_result['error']}") 

    def _calculate_orchestration_usage(self, orchestration_log: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total usage from orchestration stages."""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        # Track tool costs
        total_tool_cost = 0.0
        total_exa_searches = 0
        total_code_executions = 0
        total_data_analyses = 0
        
        # Sum up usage from all stages
        for stage in orchestration_log.get('stages', []):
            if stage.get('success') and 'usage_stats' in stage:
                usage = stage['usage_stats']
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)
            
            # Sum up tool costs
            if stage.get('success') and 'tool_costs' in stage:
                tool_costs = stage['tool_costs']
                total_tool_cost += tool_costs.get('costs', {}).get('total_tool_cost', 0.0)
                total_exa_searches += tool_costs.get('exa_searches', 0)
                total_code_executions += tool_costs.get('code_executions', 0)
                total_data_analyses += tool_costs.get('data_analyses', 0)
        
        # Calculate real costs using Grok 3 mini beta pricing
        llm_input_cost = (total_prompt_tokens / 1_000_000) * 0.30
        llm_output_cost = (total_completion_tokens / 1_000_000) * 0.50
        llm_total_cost = llm_input_cost + llm_output_cost
        
        # Total cost includes LLM + tools
        total_cost = llm_total_cost + total_tool_cost
        
        return {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens,
            'costs': {
                'llm_input_cost': llm_input_cost,
                'llm_output_cost': llm_output_cost,
                'llm_total_cost': llm_total_cost,
                'tool_cost': total_tool_cost,
                'total_cost': total_cost,
                'currency': 'USD',
                'pricing_model': 'grok-3-mini-beta + tools'
            },
            'tool_usage': {
                'exa_searches': total_exa_searches,
                'code_executions': total_code_executions,
                'data_analyses': total_data_analyses,
                'total_tool_cost': total_tool_cost
            },
            'agents_used': len(orchestration_log.get('agents_used', [])),
            'stages_completed': len([s for s in orchestration_log.get('stages', []) if s.get('success')]),
            'breakdown': {
                'stages': [
                    {
                        'stage_name': stage.get('stage_name', 'unknown'),
                        'agent_name': stage.get('agent_name', 'unknown'),
                        'agent_role': stage.get('agent_role', 'unknown'),
                        'usage': stage.get('usage_stats', {}),
                        'llm_cost': self._calculate_individual_cost(stage.get('usage_stats', {})),
                        'tool_costs': stage.get('tool_costs', {}),
                        'total_cost': (
                            self._calculate_individual_cost(stage.get('usage_stats', {})).get('total_cost', 0.0) +
                            stage.get('tool_costs', {}).get('costs', {}).get('total_tool_cost', 0.0)
                        )
                    }
                    for stage in orchestration_log.get('stages', []) if stage.get('success')
                ]
            }
        } 

    def _calculate_parallel_usage(self, worker_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate usage for enhanced parallel orchestration."""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        
        # Track tool costs
        total_tool_cost = 0.0
        total_exa_searches = 0
        total_code_executions = 0
        total_data_analyses = 0
        
        # Sum up worker usage
        for result in worker_results:
            if result.get('success') and 'usage_stats' in result:
                usage = result['usage_stats']
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)
            
            # Sum up tool costs
            if result.get('success') and 'tool_costs' in result:
                tool_costs = result['tool_costs']
                total_tool_cost += tool_costs.get('costs', {}).get('total_tool_cost', 0.0)
                total_exa_searches += tool_costs.get('exa_searches', 0)
                total_code_executions += tool_costs.get('code_executions', 0)
                total_data_analyses += tool_costs.get('data_analyses', 0)
        
        # Calculate costs
        llm_input_cost = (total_prompt_tokens / 1_000_000) * 0.30
        llm_output_cost = (total_completion_tokens / 1_000_000) * 0.50
        llm_total_cost = llm_input_cost + llm_output_cost
        total_cost = llm_total_cost + total_tool_cost
        
        return {
            'prompt_tokens': total_prompt_tokens,
            'completion_tokens': total_completion_tokens,
            'total_tokens': total_tokens,
            'costs': {
                'llm_input_cost': llm_input_cost,
                'llm_output_cost': llm_output_cost,
                'llm_total_cost': llm_total_cost,
                'tool_cost': total_tool_cost,
                'total_cost': total_cost,
                'currency': 'USD',
                'pricing_model': 'grok-3-mini-beta + tools'
            },
            'tool_usage': {
                'exa_searches': total_exa_searches,
                'code_executions': total_code_executions,
                'data_analyses': total_data_analyses,
                'total_tool_cost': total_tool_cost
            },
            'agents_used': len([r for r in worker_results if r.get('success')]),
            'breakdown': {
                'workers': [
                    {
                        'agent': result.get('agent_name', 'unknown'),
                        'usage': result.get('usage_stats', {}),
                        'llm_cost': self._calculate_individual_cost(result.get('usage_stats', {})),
                        'tool_costs': result.get('tool_costs', {}),
                        'total_cost': (
                            self._calculate_individual_cost(result.get('usage_stats', {})).get('total_cost', 0.0) +
                            result.get('tool_costs', {}).get('costs', {}).get('total_tool_cost', 0.0)
                        )
                    }
                    for result in worker_results if result.get('success')
                ]
            }
        } 

    async def _orchestrate_parallel_dynamic(self, query: str, session_id: str, active_agents: List[WorkerAgent]) -> Dict[str, Any]:
        """Enhanced parallel orchestration with dynamically spawned specialized agents."""
        logger.info(f"Using dynamic parallel orchestration with {len(active_agents)} agents...")
        
        # Separate base agents from spawned agents
        base_agents = [agent for agent in active_agents if agent not in self.spawned_agents]
        spawned_agents = [agent for agent in active_agents if agent in self.spawned_agents]
        
        if spawned_agents:
            logger.info(f"Active agents: {len(base_agents)} base + {len(spawned_agents)} spawned")
        
        # Execute all agents in parallel with specialized prompts
        worker_results = []
        timeout = self.system_config.get('task_timeout', 120)
        
        async with aiohttp.ClientSession() as session:
            # Phase 1: Parallel execution of all agents
            tasks = []
            for agent in active_agents:
                # Create specialized prompt based on agent type and focus
                agent_prompt = self._create_specialized_prompt(query, agent)
                
                task = asyncio.create_task(
                    self._execute_worker_with_timeout(agent, agent_prompt, session, timeout)
                )
                tasks.append(task)
            
            # Wait for parallel execution
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    worker_results.append({
                        'success': False,
                        'agent_name': active_agents[i].name,
                        'model': active_agents[i].model,
                        'response': None,
                        'tool_usage': [],
                        'tool_costs': {},
                        'processing_time': 0,
                        'error': str(result),
                        'agent_type': 'spawned' if active_agents[i] in spawned_agents else 'base'
                    })
                else:
                    result['agent_type'] = 'spawned' if active_agents[i] in spawned_agents else 'base'
                    worker_results.append(result)
            
            # Filter successful results
            successful_results = [r for r in worker_results if r['success']]
            
            if not successful_results:
                return {
                    'success': False,
                    'error': 'All agents failed',
                    'orchestration_log': {
                        'mode': 'parallel_dynamic',
                        'worker_results': worker_results,
                        'total_agents': len(active_agents),
                        'spawned_agents': len(spawned_agents)
                    }
                }
            
            # Phase 2: Intelligent synthesis based on agent specializations
            final_response_content = await self._synthesize_dynamic_results(successful_results, query)
            
            # Calculate total usage
            total_usage = self._calculate_parallel_usage(worker_results)
            
            return {
                'success': True,
                'final_response': final_response_content,
                'orchestration_log': {
                    'mode': 'parallel_dynamic',
                    'worker_results': worker_results,
                    'total_agents': len(active_agents),
                    'base_agents': len(base_agents),
                    'spawned_agents': len(spawned_agents),
                    'synthesis_method': 'dynamic_specialization'
                },
                'total_usage': total_usage
            }

    def _create_specialized_prompt(self, query: str, agent: WorkerAgent) -> str:
        """Create a specialized prompt based on the agent's role and focus area."""
        agent_config = agent.agent_config
        role = agent_config.get('role', 'researcher')
        focus_area = agent_config.get('focus_area', '')
        specialization = agent_config.get('specialization', '')
        template = agent_config.get('template', '')
        
        # Base prompt
        base_prompt = f"Original Query: {query}\n\n"
        
        # Add specialization based on agent type
        if template == 'comparative_researcher':
            specialized_prompt = f"""As a Comparative Research Specialist focusing on "{focus_area}", your task is to:

1. Research and analyze "{focus_area}" specifically in relation to this query
2. Gather comprehensive data, facts, and insights about "{focus_area}"
3. Use web search tools when needed for current information
4. Focus on providing detailed, accurate information that will be valuable for comparison
5. Structure your findings clearly for synthesis with other comparative analyses

Provide thorough research specifically about "{focus_area}" that addresses the query from this focused perspective."""

        elif template == 'geographic_researcher':
            specialized_prompt = f"""As a Geographic Research Specialist focusing on "{focus_area}", your task is to:

1. Research information specifically related to "{focus_area}" and this query
2. Gather region-specific data, policies, trends, and insights
3. Use web search tools for current regional information when needed
4. Consider cultural, economic, and political factors specific to "{focus_area}"
5. Provide context that's unique to this geographic region

Focus exclusively on "{focus_area}" and provide comprehensive regional insights that address the query."""

        elif template == 'coder_specialist':
            specialized_prompt = f"""As a Coding and Visualization Specialist, your task is to:

1. Analyze the query for any computational or coding requirements
2. Write Python code to address calculation, data processing, or visualization needs
3. Use the code_executor tool to run and test your code
4. Create clear, well-documented code with explanations
5. If visualization is needed, generate appropriate charts or graphs

Focus on the technical and computational aspects of this query. Write and execute code to provide concrete, calculated results."""

        elif role == 'researcher':
            specialized_prompt = f"""As the Research Specialist, your task is to:

1. Gather comprehensive, accurate information using available tools when needed
2. Focus on factual data and current information
3. Use web search for recent developments or current data
4. Use code execution for any calculations or data processing
5. Provide well-researched, evidence-based information

Focus on thorough research that directly addresses this query with accurate, up-to-date information."""

        elif role == 'synthesizer':
            specialized_prompt = f"""As the Content Synthesizer, your task is to:

1. Create a comprehensive, well-structured response to this query
2. Organize information logically and clearly
3. Ensure the response is accessible and complete
4. Address all aspects of the original query
5. Provide actionable insights where appropriate

Create a polished, comprehensive response that fully addresses the user's query."""

        else:
            # Default prompt for other roles
            specialized_prompt = f"""Your task is to provide a high-quality response to this query using your specialized capabilities.

Focus on delivering accurate, helpful information that addresses the user's needs."""

        return base_prompt + specialized_prompt

    async def _synthesize_dynamic_results(self, successful_results: List[Dict[str, Any]], query: str) -> str:
        """Intelligently synthesize results from dynamically spawned specialized agents."""
        if len(successful_results) == 1:
            return successful_results[0]['response']
        
        # Categorize results by agent type and specialization
        comparative_results = []
        geographic_results = []
        coding_results = []
        general_results = []
        
        for result in successful_results:
            agent_name = result['agent_name']
            response = result['response']
            
            if 'comparative_researcher' in agent_name:
                comparative_results.append(f"Comparative Analysis: {response}")
            elif 'geographic_researcher' in agent_name:
                geographic_results.append(f"Geographic Analysis: {response}")
            elif 'coder_specialist' in agent_name:
                coding_results.append(f"Technical Analysis: {response}")
            else:
                general_results.append(f"General Analysis: {response}")
        
        # Combine results intelligently
        synthesized_parts = []
        
        if comparative_results:
            synthesized_parts.append("## Comparative Analysis\n" + "\n\n".join(comparative_results))
        
        if geographic_results:
            synthesized_parts.append("## Regional Perspectives\n" + "\n\n".join(geographic_results))
        
        if coding_results:
            synthesized_parts.append("## Technical Implementation\n" + "\n\n".join(coding_results))
        
        if general_results:
            synthesized_parts.append("## Comprehensive Analysis\n" + "\n\n".join(general_results))
        
        if len(synthesized_parts) > 1:
            final_response = f"# Comprehensive Response to: {query}\n\n" + "\n\n".join(synthesized_parts)
        else:
            final_response = synthesized_parts[0] if synthesized_parts else successful_results[0]['response']
        
        return final_response

    async def _execute_intelligent_planning(self, planner: WorkerAgent, query: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Execute intelligent planning stage with JSON output."""
        planning_prompt = f"""You are a Master Planner AI that creates structured execution plans for complex queries.

Query: "{query}"

Analyze this query and create a step-by-step execution plan in JSON format. Consider:
1. What information needs to be gathered
2. What specialized agents are needed
3. Dependencies between tasks
4. Final synthesis requirements

The plan should consist of 'stages'. Each stage needs:
- 'stage_id': A unique identifier (e.g., "research_us_policy")
- 'prompt': The specific instruction for the agent in this stage
- 'agent_template': MUST be one of: "comparative_researcher", "geographic_researcher", "coder_specialist", "synthesizer"
- 'focus_area': The specific topic for this agent (e.g., "US Policy")
- 'dependencies': A list of 'stage_id's that must be completed before this stage can start. Use [] for initial stages.
- 'synthesis_input': boolean, if true, the output will be fed into the final synthesis

VALID AGENT TEMPLATES:
- "comparative_researcher": For comparing subjects or gathering focused research
- "geographic_researcher": For region-specific analysis
- "coder_specialist": For coding, calculations, and visualizations
- "synthesizer": For final synthesis and comprehensive responses

Example JSON format:
{{
  "plan": [
    {{ "stage_id": "research_us", "prompt": "Research US policy on X", "agent_template": "comparative_researcher", "focus_area": "US", "dependencies": [], "synthesis_input": true }},
    {{ "stage_id": "research_cn", "prompt": "Research China policy on X", "agent_template": "comparative_researcher", "focus_area": "China", "dependencies": [], "synthesis_input": true }},
    {{ "stage_id": "final_synthesis", "prompt": "Synthesize the research findings", "agent_template": "synthesizer", "focus_area": "final_answer", "dependencies": ["research_us", "research_cn"], "synthesis_input": false }}
  ]
}}

Now generate the JSON execution plan for the query. Respond with ONLY the JSON, no additional text."""
        
        return await self._execute_agent_stage(planner, planning_prompt, session, "Intelligent Planning")

    def _parse_execution_plan(self, planning_response: str) -> Dict[str, Any]:
        """Parse the JSON execution plan from the planner's response."""
        import json
        import re
        
        try:
            # Clean the response to extract JSON
            cleaned_response = planning_response.strip()
            
            # Try to find JSON in the response using better regex
            json_match = re.search(r'\{.*?"plan".*?\[.*?\].*?\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Try to extract everything between first { and last }
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = cleaned_response[start_idx:end_idx+1]
                else:
                    json_str = cleaned_response
            
            # Clean up common JSON issues
            json_str = json_str.replace('\n', ' ').replace('\t', ' ')
            # Fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Parse JSON
            plan = json.loads(json_str)
            
            # Validate plan structure
            if not isinstance(plan, dict) or 'plan' not in plan:
                logger.warning("Invalid plan structure - missing 'plan' key")
                return None
            
            # Validate each stage
            valid_templates = ['comparative_researcher', 'geographic_researcher', 'coder_specialist', 'synthesizer']
            for stage in plan['plan']:
                required_fields = ['stage_id', 'prompt', 'agent_template', 'focus_area', 'dependencies', 'synthesis_input']
                if not all(field in stage for field in required_fields):
                    logger.warning(f"Invalid stage structure: {stage}")
                    return None
                
                # Validate agent template
                if stage['agent_template'] not in valid_templates:
                    logger.warning(f"Invalid agent template '{stage['agent_template']}' in stage {stage['stage_id']}")
                    # Try to fix common template names
                    template_mapping = {
                        'data_researcher': 'comparative_researcher',
                        'comparative_analyst': 'comparative_researcher',
                        'visualization_specialist': 'coder_specialist',
                        'researcher': 'comparative_researcher'
                    }
                    if stage['agent_template'] in template_mapping:
                        stage['agent_template'] = template_mapping[stage['agent_template']]
                        logger.info(f"Fixed agent template to: {stage['agent_template']}")
                    else:
                        logger.error(f"Cannot fix invalid agent template: {stage['agent_template']}")
                        return None
            
            logger.info(f"Successfully parsed execution plan with {len(plan['plan'])} stages")
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON plan: {e}")
            logger.error(f"Response was: {planning_response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing execution plan: {e}")
            return None

    def _display_execution_plan(self, execution_plan: Dict[str, Any]) -> None:
        """Display the execution plan to the user."""
        print("\n" + "="*60)
        print("GROK HEAVY - Dynamic Execution Plan")
        print("="*60)
        
        plan_stages = execution_plan['plan']
        
        for i, stage in enumerate(plan_stages, 1):
            stage_id = stage['stage_id']
            agent_template = stage['agent_template']
            focus_area = stage['focus_area']
            dependencies = stage['dependencies']
            
            print(f"\nStage {i}: {stage_id} ({agent_template})")
            print(f"   Focus: {focus_area}")
            if dependencies:
                print(f"   Depends on: {', '.join(dependencies)}")
            else:
                print(f"   Dependencies: None (can start immediately)")
        
        print("\nExecuting plan...")
        print("-" * 60)
        print()

    async def _execute_dynamic_plan(self, execution_plan: Dict[str, Any], query: str, active_agents: List[WorkerAgent], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Execute the dynamic plan with dependency management."""
        plan_stages = execution_plan['plan']
        completed_stages = []
        stage_outputs = {}
        stage_results = []
        execution_log = []
        
        max_iterations = len(plan_stages) + 2  # Safety valve
        iteration = 0
        
        while len(completed_stages) < len(plan_stages) and iteration < max_iterations:
            iteration += 1
            
            # Find stages that are ready to execute
            ready_stages = []
            for stage in plan_stages:
                stage_id = stage['stage_id']
                if stage_id not in completed_stages:
                    dependencies = stage['dependencies']
                    if all(dep in completed_stages for dep in dependencies):
                        ready_stages.append(stage)
            
            if not ready_stages:
                error_msg = f"No ready stages found. Completed: {completed_stages}, Remaining: {[s['stage_id'] for s in plan_stages if s['stage_id'] not in completed_stages]}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'execution_log': execution_log,
                    'stage_results': stage_results
                }
            
            # Execute ready stages in parallel
            logger.info(f"Executing {len(ready_stages)} parallel stages: {[s['stage_id'] for s in ready_stages]}")
            print(f"\n>>> EXECUTING [{', '.join([s['stage_id'] for s in ready_stages])}] in parallel...")
            for stage in ready_stages:
                print(f"   - Running '{stage['stage_id']}' with agent {stage['agent_template']}...")
            
            # Create tasks for parallel execution
            tasks = []
            for stage in ready_stages:
                task = asyncio.create_task(self._execute_plan_stage(stage, query, stage_outputs, active_agents, session))
                tasks.append((stage, task))
            
            # Wait for all tasks to complete
            for stage, task in tasks:
                try:
                    result = await task
                    stage_results.append(result)
                    
                    if result['success']:
                        completed_stages.append(stage['stage_id'])
                        stage_outputs[stage['stage_id']] = result['response']
                        logger.info(f"Completed stage: {stage['stage_id']}")
                        print(f"\n[SUCCESS] {stage['stage_id']} completed.")
                    else:
                        logger.error(f"Failed stage: {stage['stage_id']} - {result.get('error', 'Unknown error')}")
                        print(f"\n[FAILED] {stage['stage_id']}: {result.get('error', 'Unknown error')}")
                        return {
                            'success': False,
                            'error': f"Stage {stage['stage_id']} failed: {result.get('error', 'Unknown error')}",
                            'execution_log': execution_log,
                            'stage_results': stage_results
                        }
                except Exception as e:
                    logger.error(f"Exception in stage {stage['stage_id']}: {str(e)}")
                    return {
                        'success': False,
                        'error': f"Exception in stage {stage['stage_id']}: {str(e)}",
                        'execution_log': execution_log,
                        'stage_results': stage_results
                    }
            
            execution_log.append({
                'iteration': iteration,
                'executed_stages': [s['stage_id'] for s in ready_stages],
                'completed_total': len(completed_stages),
                'remaining': len(plan_stages) - len(completed_stages)
            })
        
        # Find the final response (from the last stage or synthesis stage)
        final_response = None
        for stage in reversed(plan_stages):
            if stage['stage_id'] in stage_outputs:
                final_response = stage_outputs[stage['stage_id']]
                break
        
        if not final_response:
            final_response = "Plan execution completed but no final response found."
        
        print(f"\n[SUCCESS] ALL STAGES COMPLETED successfully!")
        print("-" * 60)
        
        return {
            'success': True,
            'final_response': final_response,
            'execution_log': execution_log,
            'stage_results': stage_results
        }

    async def _execute_plan_stage(self, stage: Dict[str, Any], query: str, stage_outputs: Dict[str, str], active_agents: List[WorkerAgent], session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Execute a single stage of the dynamic plan."""
        stage_id = stage['stage_id']
        agent_template = stage['agent_template']
        focus_area = stage['focus_area']
        base_prompt = stage['prompt']
        dependencies = stage['dependencies']
        
        # Find or spawn the appropriate agent
        agent = self._find_or_spawn_agent(agent_template, focus_area, active_agents)
        if not agent:
            return {
                'success': False,
                'error': f"Could not find or spawn agent for template: {agent_template}",
                'stage_id': stage_id,
                'agent_template': agent_template
            }
        
        # Construct the full prompt with context from dependencies
        full_prompt = f"Original Query: {query}\n\n"
        
        # Add dependency outputs as context
        if dependencies:
            full_prompt += "Context from Previous Stages:\n"
            for dep_id in dependencies:
                if dep_id in stage_outputs:
                    full_prompt += f"\n**[{dep_id}]:**\n{stage_outputs[dep_id]}\n"
            full_prompt += "\n"
        
        # Add the specific stage prompt
        full_prompt += f"Your Task: {base_prompt}\n\n"
        
        # Add specialization based on agent template
        if agent_template == "synthesizer":
            full_prompt += "As the Synthesizer, combine all the provided context into a comprehensive, well-structured final response that fully addresses the original query."
        elif agent_template == "comparative_researcher":
            full_prompt += f"As a Comparative Researcher focusing on '{focus_area}', provide detailed analysis and research specifically about this subject."
        elif agent_template == "coder_specialist":
            full_prompt += "As a Coding Specialist, write and execute Python code to address any computational requirements. Use the code_executor tool as needed."
        else:
            full_prompt += f"Focus your expertise on '{focus_area}' and provide high-quality analysis."
        
        # Execute the stage
        try:
            result = await self._execute_agent_stage(agent, full_prompt, session, f"Stage: {stage_id}")
            result['stage_id'] = stage_id
            result['agent_template'] = agent_template
            result['focus_area'] = focus_area
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'stage_id': stage_id,
                'agent_template': agent_template,
                'focus_area': focus_area
            }

    def _find_or_spawn_agent(self, agent_template: str, focus_area: str, active_agents: List[WorkerAgent]) -> Optional[WorkerAgent]:
        """Find an existing agent or spawn a new one for the given template."""
        # First, try to find an existing agent with the right role
        template_to_role = {
            'synthesizer': 'synthesizer',
            'comparative_researcher': 'researcher',
            'coder_specialist': 'researcher',
            'geographic_researcher': 'researcher'
        }
        
        target_role = template_to_role.get(agent_template)
        if target_role:
            for agent in active_agents:
                if hasattr(agent, 'agent_config') and agent.agent_config.get('role') == target_role:
                    return agent
        
        # If no existing agent found, try to spawn one
        spawned_agent = self._spawn_agent(agent_template, focus_area, f"Dynamic plan execution for {agent_template}")
        if spawned_agent:
            return spawned_agent
        
        # Fallback to any available agent
        if active_agents:
            logger.warning(f"Using fallback agent for template {agent_template}")
            return active_agents[0]
        
        return None 
    async def _orchestrate_sequential_dynamic(self, query: str, session_id: str, enable_critique: bool, active_agents: List[WorkerAgent]) -> Dict[str, Any]:
        """Advanced sequential orchestration with dynamic planning and execution."""
        logger.info("Starting dynamic sequential orchestration with intelligent planning...")
        
        orchestration_log = {
            'mode': 'sequential_dynamic',
            'stages': [],
            'agents_used': [],
            'plan': None,
            'plan_execution': []
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                # Stage 1: Intelligent Planning
                logger.info("Stage 1: Intelligent Planning...")
                planner = self._find_agent_by_role('planner') or active_agents[0]
                
                planning_result = await self._execute_intelligent_planning(planner, query, session)
                orchestration_log['stages'].append(planning_result)
                orchestration_log['agents_used'].append(planner.name)
                
                if not planning_result['success']:
                    logger.error("Planning stage failed, falling back to parallel mode")
                    return await self._orchestrate_parallel_dynamic(query, session_id, active_agents)
                
                # Parse the execution plan
                execution_plan = self._parse_execution_plan(planning_result['response'])
                orchestration_log['plan'] = execution_plan
                
                if not execution_plan or not execution_plan.get('plan'):
                    logger.warning("No valid execution plan received, falling back to parallel mode")
                    return await self._orchestrate_parallel_dynamic(query, session_id, active_agents)
                
                # Stage 2: Execute the Plan
                logger.info(f"Stage 2: Executing Dynamic Plan ({len(execution_plan['plan'])} stages)...")
                self._display_execution_plan(execution_plan)
                
                plan_execution_result = await self._execute_dynamic_plan(execution_plan, query, active_agents, session)
                orchestration_log['plan_execution'] = plan_execution_result['execution_log']
                orchestration_log['stages'].extend(plan_execution_result['stage_results'])
                
                if not plan_execution_result['success']:
                    return {
                        'success': False,
                        'error': plan_execution_result['error'],
                        'orchestration_log': orchestration_log
                    }
                
                # Get the final response from the plan execution
                final_response = plan_execution_result['final_response']
                
                # Calculate total usage
                total_usage = self._calculate_orchestration_usage(orchestration_log)
                
                return {
                    'success': True,
                    'final_response': final_response,
                    'orchestration_log': orchestration_log,
                    'total_usage': total_usage
                }
                
            except Exception as e:
                logger.error(f"Dynamic sequential orchestration error: {str(e)}")
                return {
                    'success': False,
                    'error': str(e),
                    'orchestration_log': orchestration_log
                }
