# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

"Grok Heavy" is a sophisticated multi-agent AI system that processes queries through parallel agent collaboration and intelligent orchestration. The system features two primary execution modes:

1. **Sequential Dynamic Mode** - Uses intelligent planning to create execution stages with dependencies, then executes them systematically
2. **Parallel Dynamic Mode** - Runs multiple research agents in parallel and synthesizes their responses

## Architecture Components

### Core Classes
- **SupervisorAgent** (`supervisor.py`) - Central orchestrator managing the entire workflow
- **WorkerAgent** (`worker.py`) - Individual agents that execute LLM calls and tool usage
- **ToolExecutor** (`tools.py`) - Handles web search, code execution, and data analysis tools

### Execution Flow
```
User Query → SupervisorAgent → Dynamic Planning (sequential mode) OR Parallel Execution
                ↓
WorkerAgents process tasks using specialized roles and tools
                ↓
Intelligent Synthesis → Final Response + Markdown Report
```

## Key Development Commands

### Running the System
```bash
# Basic query execution
python main.py "Your query here"

# With verbose logging to see individual agent responses
python main.py "Your query here" --verbose

# Debug mode to see raw JSON planning responses and parsing steps
python main.py "Your query here" --log-level DEBUG

# Interactive mode for continuous queries
python main.py
```

### Testing
```bash
# Run core functionality tests
python test_core_fixes.py

# Test JSON parsing and orchestration logic
python test_orchestrator.py

# Test system integration
python test_system.py

# Test specific fixes
python test_fixes.py
```

### Configuration
- Main config: `config.yaml` - Contains agent definitions, roles, and system settings
- Environment: `.env` file with `OPENROUTER_API_KEY` required

## Critical Architecture Details

### Agent Role System
The system uses specialized agent roles defined in `config.yaml`:
- **planner** - Creates execution plans and breaks down complex queries
- **researcher** - Gathers information using tools (web search, code execution)
- **critic** - Reviews work for accuracy and quality
- **synthesizer** - Combines information into coherent responses
- **coder_specialist** - Handles programming and visualization tasks

### Dynamic Planning System
The sequential mode creates JSON execution plans with:
- `stage_id` - Unique identifier for each execution stage
- `agent_template` - Specifies which type of agent to use
- `dependencies` - Array of stage IDs that must complete first
- `synthesis_input` - Boolean indicating if output feeds into final synthesis

### JSON Parsing Robustness
The system includes robust JSON parsing to handle LLM output issues:
- **FIXED**: Uses greedy regex matching (`r'\{.*"plan".*\}'`) to capture complete JSON objects
- Removes escape characters that cause "Invalid \escape" errors
- Fixes string-wrapped arrays and booleans
- Handles trailing commas and malformed quotes
- Includes DEBUG logging to diagnose parsing failures

### Tool Integration
Agents can use tools via XML-style tags in their responses:
```xml
<tool name="web_search" query="search terms"/>
<tool name="code_executor" code="python code here"/>
<tool name="data_explorer" query="analysis request">CSV data</tool>
```

### Visualization Handling
**Critical:** The coder_specialist agents are explicitly instructed to:
- NEVER use `plt.show()` (causes system hangs)
- Always save plots with `plt.savefig('filename.png')`
- Report saved filenames in responses

## Fallback Mechanisms

The system has robust error handling:
1. If sequential planning fails → falls back to parallel mode
2. If JSON parsing fails → extensive cleaning and retry logic
3. If individual agents fail → continues with successful agents
4. If synthesis fails → returns longest individual response

## Output and Logging

### Session Logs
- JSON logs saved to `logs/session_YYYYMMDD_HHMMSS.json`
- Markdown reports generated in `logs/run_YYYYMMDD_HHMMSS/final_report.md`
- Reports automatically embed generated visualizations

### Debug Information
Use `--log-level DEBUG` to see:
- Raw planner JSON responses before parsing
- JSON cleaning steps and attempts
- Detailed error messages and fallback triggers

## Development Patterns

### Adding New Agent Templates
1. Define in `config.yaml` under `dynamic_spawning.agent_templates`
2. Add role-specific system prompt in `worker.py._get_system_prompt()`
3. Update template mapping in `supervisor.py._find_or_spawn_agent()`

### Adding New Tools
1. Implement in `tools.py.ToolExecutor`
2. Add parsing logic in `tools.py.parse_tool_requests()`
3. Update agent system prompts to describe new tool

### Debugging JSON Parsing Issues
1. Run with `--log-level DEBUG`
2. Check logs for "RAW PLANNER RESPONSE" and "EXTRACTED JSON STRING"
3. The regex `r'\{.*"plan".*\}'` uses greedy matching to capture complete JSON objects
4. Update cleaning regex in `supervisor.py._parse_execution_plan()` if needed

## Common Issues and Solutions

### Fixed Issues (v2025-01-11)

**Issue:** JSON parsing truncates multi-stage plans
**Solution:** ✅ **FIXED** - Changed regex from `r'\{.*?"plan".*?\}'` to `r'\{.*"plan".*\}'` (greedy matching)
- **Impact**: Multi-stage plans now parse correctly instead of being truncated at first `}`
- **Location**: `supervisor.py:_parse_execution_plan()` line 1974

**Issue:** Parallel mode uses only 1 agent instead of multiple
**Solution:** ✅ **FIXED** - Added logic to ensure at least 2 agents are used when available
- **Impact**: Parallel execution now uses multiple agents for better results
- **Location**: `supervisor.py:_orchestrate_parallel_dynamic()` lines 1699-1701

### Remaining Common Issues

**Issue:** Code execution hangs on visualizations  
**Solution:** Ensure coder prompts explicitly prohibit `plt.show()` and require `plt.savefig()`

**Issue:** JSON parsing fails with "Invalid \escape"
**Solution:** Enhanced backslash removal in `_parse_execution_plan()` handles this

**Issue:** Sequential mode falls back to parallel unexpectedly
**Solution:** Enable DEBUG logging to see exact planner output and parsing failures

## Testing and Verification

### Regex Fix Verification
The JSON parsing fix was verified using direct regex testing:
```python
# Test multi-stage plan parsing
test_response = '''
{
  "plan": [
    {"stage_id": "stage1", "prompt": "Task 1", "agent_template": "researcher", "dependencies": [], "synthesis_input": true},
    {"stage_id": "stage2", "prompt": "Task 2", "agent_template": "researcher", "dependencies": ["stage1"], "synthesis_input": true}
  ]
}
'''

# Old regex (broken): captures 140 chars, truncated JSON
old_match = re.search(r'\{.*?"plan".*?\}', test_response, re.DOTALL)

# New regex (fixed): captures 282 chars, complete JSON
new_match = re.search(r'\{.*"plan".*\}', test_response, re.DOTALL)
```

### Agent Selection Fix Verification
The parallel mode fix ensures multiple agents are used:
```python
# Before fix: Could use only 1 agent
researcher_agents = suitable_agents[:3] if len(suitable_agents) > 1 else suitable_agents

# After fix: Ensures at least 2 agents when available
if len(researcher_agents) < 2 and len(active_agents) > 1:
    researcher_agents = active_agents[:3]
```