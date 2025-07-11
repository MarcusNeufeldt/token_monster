#!/usr/bin/env python3
"""
Grok Heavy - Multi-Agent AI System
Main CLI interface for the framework-free multi-agent system.
"""

import argparse
import asyncio
import logging
import os
import sys
import yaml
from dotenv import load_dotenv
from supervisor import SupervisorAgent

# Setup logging function - will be called with dynamic level
def setup_logging(level_name: str = 'INFO'):
    """Setup logging with the specified level."""
    log_level = getattr(logging, level_name.upper(), logging.INFO)
    
    # Remove all handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('grok_heavy.log', mode='w')  # Overwrite log each run
        ]
    )

# Initial setup with INFO level (will be overridden by argparse)
setup_logging('INFO')

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def load_api_key() -> str:
    """Load API key from environment variables."""
    load_dotenv()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment variables")
        print("Error: Please set your OPENROUTER_API_KEY in the .env file")
        sys.exit(1)
    
    # Also check for EXA_API_KEY (optional but recommended for web search)
    exa_key = os.getenv('EXA_API_KEY')
    if not exa_key:
        logger.warning("EXA_API_KEY not found - web search functionality may be limited")
        print("Warning: EXA_API_KEY not found - web search functionality may be limited")
    
    return api_key


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Grok Heavy - Multi-Agent AI System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What is the capital of France?"
  python main.py "Explain quantum computing" --verbose
  python main.py "Write a Python function to sort a list" --config custom_config.yaml

For more information, visit: https://github.com/your-repo/grok-heavy
        """
    )
    
    parser.add_argument(
        'query',
        help='The question or prompt to process through the multi-agent system'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed deliberation logs and individual agent responses'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Grok Heavy v1.0.0'
    )
    
    return parser


async def main():
    """Main application entry point."""
    # Setup argument parser
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Configure logging level with proper setup
    setup_logging(args.log_level)
    
    # Load configuration and API key
    config = load_config(args.config)
    api_key = load_api_key()
    
    # Display startup information
    print("🚀 GROK HEAVY - Multi-Agent AI System")
    print("="*50)
    print(f"Query: {args.query}")
    print(f"Config: {args.config}")
    print(f"Verbose: {args.verbose}")
    
    # Initialize supervisor
    try:
        supervisor = SupervisorAgent(config, api_key)
        print(f"Initialized {len(supervisor.workers)} worker agents")
        
        # Process the query
        result = await supervisor.process_query(args.query, args.verbose)
        
        # Return appropriate exit code
        sys.exit(0 if result['success'] else 1)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\nFatal Error: {str(e)}")
        sys.exit(1)


def interactive_mode():
    """Run in interactive mode for continuous queries."""
    print("🚀 GROK HEAVY - Interactive Mode")
    print("="*50)
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'help' for available commands")
    print()
    
    # Load configuration and API key
    config = load_config()
    api_key = load_api_key()
    
    # Initialize supervisor
    supervisor = SupervisorAgent(config, api_key)
    print(f"Initialized {len(supervisor.workers)} worker agents\n")
    
    verbose = False
    
    while True:
        try:
            user_input = input("🤖 Query: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("""
Available commands:
  help      - Show this help message
  verbose   - Toggle verbose output
  status    - Show system status
  exit/quit - Exit interactive mode
  
Or enter any question to get a multi-agent response.
                """)
                continue
            elif user_input.lower() == 'verbose':
                verbose = not verbose
                print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
                continue
            elif user_input.lower() == 'status':
                print(f"Active workers: {len(supervisor.workers)}")
                for worker in supervisor.workers:
                    print(f"  - {worker.name} ({worker.model})")
                continue
            elif not user_input:
                continue
            
            # Process the query
            result = asyncio.run(supervisor.process_query(user_input, verbose))
            
        except KeyboardInterrupt:
            print("\nUse 'exit' or 'quit' to end the session.")
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"Error: {str(e)}")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments provided, run in interactive mode
        interactive_mode()
    else:
        # Arguments provided, run in CLI mode
        asyncio.run(main()) 