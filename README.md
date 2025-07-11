# 🚀 Grok Heavy - Multi-Agent AI System

A framework-free, parallel multi-agent AI system that enhances response quality through collaborative deliberation and consensus-building.

## 🌟 Overview

Grok Heavy is a sophisticated multi-agent AI system designed to improve the accuracy, reliability, and creativity of AI responses by deploying multiple diverse AI agents in parallel. These agents independently process queries and then collaborate through a structured deliberation process to form high-quality consensus answers.

### Key Features

- **🔄 Parallel Processing**: Multiple agents work simultaneously for faster results
- **🧠 Multi-Agent Grok Integration**: Leverages multiple instances of xAI's Grok 3 mini beta with high reasoning effort
- **🛠️ Tool Integration**: Built-in web search and code execution capabilities  
- **📊 Deliberation Layer**: Advanced synthesis and consensus mechanisms
- **🔍 Transparency**: Detailed logging and verbose mode for analysis
- **⚡ Framework-Free**: Built from scratch in Python for maximum control
- **🔧 Extensible**: Easy to add new models, agents, and tools

## 🏗️ Architecture

```
User Input (CLI) --> Supervisor Agent
                     |
                     v
Async Task Pool:  [Worker 1 (Grok)] <--> [Worker 2 (Grok)] <--> [Worker 3 (Grok)]
                     |                            |                            |
                     v (Tool Request)             v (Tool Request)             v (Tool Request)
                   [Tool Executor (Search, Code)] -> Returns data to worker
                     |                            |                            |
                     v (Final Output)             v (Final Output)             v (Final Output)
Supervisor Collects -> Deliberation Layer (Synthesizing LLM Call)
                           |
                           v
                 Output Aggregator --> Final Response (stdout) + Logs (file)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenRouter API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd token_monster
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment**
   ```bash
   # Create .env file with your OpenRouter API key
   echo "OPENROUTER_API_KEY=your_api_key_here" > .env
   ```

4. **Run your first query**
   ```bash
   python main.py "What is the capital of France?"
   ```

### Interactive Mode

For continuous usage, run without arguments:

```bash
python main.py
```

This launches interactive mode with commands:
- `help` - Show available commands
- `verbose` - Toggle detailed logging
- `status` - Show system status
- `exit`/`quit` - Exit the session

## 💡 Usage Examples

### Basic Query
```bash
python main.py "Explain quantum computing in simple terms"
```

### Verbose Mode (shows individual agent responses)
```bash
python main.py "Write a Python function to calculate fibonacci numbers" --verbose
```

### Custom Configuration
```bash
python main.py "What are the latest developments in AI?" --config custom_config.yaml
```

### Complex Queries with Tools
```bash
python main.py "Search for recent news about SpaceX and summarize the key developments"
```

## 🔧 Configuration

The system is configured via `config.yaml`:

```yaml
# Agent Configuration
agents:
  - name: "grok_agent_1"
    model: "x-ai/grok-3-mini-beta"
    description: "xAI's Grok model with high reasoning effort - Agent 1"
    reasoning_effort: "high"
    
  - name: "grok_agent_2"
    model: "x-ai/grok-3-mini-beta"
    description: "xAI's Grok model with high reasoning effort - Agent 2"
    reasoning_effort: "high"
    
  - name: "grok_agent_3"
    model: "x-ai/grok-3-mini-beta"
    description: "xAI's Grok model with high reasoning effort - Agent 3"
    reasoning_effort: "high"

# System Settings
system:
  max_parallel_workers: 3
  task_timeout: 60
  save_logs: true
  logs_directory: "logs"

# Tool Configuration
tools:
  web_search:
    enabled: true
    max_results: 5
    
  code_executor:
    enabled: true
    timeout: 10
    sandbox: true
```

## 🛠️ Available Tools

### Web Search
Agents can search the web using Exa AI:
```xml
<tool name="web_search" query="latest AI developments 2024"/>
```

### Code Execution
Agents can execute Python code safely:
```xml
<tool name="code_executor" code="print('Hello, World!')"/>
```

## 📊 Logging and Analysis

### Session Logs
Each query generates a detailed JSON log saved to `logs/session_YYYYMMDD_HHMMSS.json` containing:
- Individual agent responses
- Tool usage details
- Processing times
- Deliberation process
- Final synthesized result

### Verbose Output
Use `--verbose` flag to see:
- Individual agent responses
- Processing times
- Tool usage statistics
- Deliberation summary

## 🔄 How It Works

1. **Query Distribution**: The Supervisor distributes the user query to all configured worker agents in parallel

2. **Parallel Processing**: Each worker agent:
   - Processes the query using its assigned LLM
   - Can use tools (web search, code execution) as needed
   - Returns its response and tool usage log

3. **Deliberation**: The Supervisor collects successful responses and uses a synthesis model to:
   - Analyze all agent responses
   - Identify the most accurate information
   - Synthesize a single, high-quality response

4. **Output**: The final synthesized response is presented to the user, with optional detailed logs

## 🎯 Performance Targets

- **Response Time**: < 30 seconds for 3 agents on standard queries
- **Accuracy Improvement**: > 20% over best single agent
- **Fault Tolerance**: Graceful handling of individual agent failures
- **Scalability**: Easy addition of new agents via configuration

## 🛡️ Security Features

- **Sandboxed Code Execution**: Code runs in isolated subprocess
- **API Key Security**: Keys loaded from environment variables
- **Input Validation**: Comprehensive error handling and validation
- **Timeout Protection**: Prevents runaway processes

## 🔧 Extending the System

### Adding New Models
1. Add model configuration to `config.yaml`:
```yaml
agents:
  - name: "new_agent"
    model: "provider/model-name"
    description: "Description of the model"
```

### Adding New Tools
1. Implement tool in `tools.py`:
```python
async def new_tool(self, param: str) -> Dict[str, Any]:
    # Tool implementation
    return {"success": True, "result": result}
```

2. Update tool parsing in `parse_tool_requests()`

### Custom Deliberation
Modify the synthesis prompt in `supervisor.py` `_deliberate()` method for specialized consensus algorithms.

## 📝 Command Line Options

```bash
usage: main.py [-h] [--verbose] [--config CONFIG] [--log-level {DEBUG,INFO,WARNING,ERROR}] [--version] query

Grok Heavy - Multi-Agent AI System

positional arguments:
  query                 The question or prompt to process

options:
  -h, --help            Show help message
  --verbose, -v         Show detailed deliberation logs
  --config CONFIG, -c   Path to configuration file (default: config.yaml)
  --log-level {DEBUG,INFO,WARNING,ERROR}
                        Set logging level (default: INFO)
  --version             Show program version
```

## 🚦 Error Handling

The system includes comprehensive error handling for:
- API failures and timeouts
- Individual agent failures
- Tool execution errors
- Configuration issues
- Network connectivity problems

Failed agents don't prevent the system from proceeding with successful agents.

## 📈 Monitoring

Monitor system performance through:
- Session logs with processing times
- Individual agent success rates
- Tool usage statistics
- API call patterns and costs

## 🤝 Contributing

This is an open-source project designed for collaboration:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (when available)
python -m pytest tests/

# Check code quality
flake8 *.py
```

## 📜 License

This project is released under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by xAI's Grok advancements and AI deliberation research
- Built for the AI developer and researcher community
- Framework-free approach for maximum transparency and learning value

---

**Ready to enhance your AI interactions with multi-agent collaboration? Get started now!** 🚀 