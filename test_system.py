#!/usr/bin/env python3
"""
Test script for Grok Heavy Multi-Agent AI System.
Verifies system functionality and configuration.
"""

import asyncio
import os
import sys
import yaml
from dotenv import load_dotenv
from supervisor import SupervisorAgent


async def test_basic_functionality():
    """Test basic system functionality."""
    print("🧪 Testing Grok Heavy System")
    print("="*40)
    
    # Test 1: Configuration loading
    print("1. Testing configuration loading...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Configuration loading failed: {e}")
        return False
    
    # Test 2: Environment variables
    print("2. Testing environment variables...")
    load_dotenv()
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print("   ✓ API key found")
    else:
        print("   ✗ API key not found in .env file")
        return False
    
    # Test 3: Agent initialization
    print("3. Testing agent initialization...")
    try:
        supervisor = SupervisorAgent(config, api_key)
        agent_count = len(supervisor.workers)
        print(f"   ✓ Initialized {agent_count} worker agents")
        
        for worker in supervisor.workers:
            print(f"      - {worker.name} ({worker.model})")
    except Exception as e:
        print(f"   ✗ Agent initialization failed: {e}")
        return False
    
    # Test 4: Simple query test
    print("4. Testing simple query processing...")
    try:
        test_query = "What is 2 + 2?"
        result = await supervisor.process_query(test_query, verbose=False)
        
        if result['success']:
            print("   ✓ Query processed successfully")
            print(f"   Response length: {len(result['final_response'])} characters")
            print(f"   Processing time: {result['processing_time']:.2f} seconds")
            print(f"   Successful workers: {len([r for r in result['worker_results'] if r['success']])}")
        else:
            print(f"   ✗ Query processing failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"   ✗ Query test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! System is ready to use.")
    return True


async def test_with_tools():
    """Test system with tool usage."""
    print("\n🔧 Testing Tool Integration")
    print("="*40)
    
    load_dotenv()
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    supervisor = SupervisorAgent(config, api_key)
    
    # Test with a query that might trigger web search
    test_query = "What are the latest developments in quantum computing? Please search for recent information."
    
    print(f"Query: {test_query}")
    print("Processing...")
    
    result = await supervisor.process_query(test_query, verbose=True)
    
    if result['success']:
        print("✓ Tool-enabled query processed successfully")
        
        # Check for tool usage
        tool_usage_count = 0
        for worker_result in result['worker_results']:
            if worker_result['success'] and worker_result['tool_usage']:
                tool_usage_count += len(worker_result['tool_usage'])
        
        print(f"   Total tool calls made: {tool_usage_count}")
    else:
        print(f"✗ Tool test failed: {result['error']}")


def run_diagnostics():
    """Run system diagnostics."""
    print("🔍 System Diagnostics")
    print("="*40)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("⚠️  Warning: Python 3.10+ recommended")
    else:
        print("✓ Python version OK")
    
    # Check required modules
    required_modules = [
        'aiohttp', 'yaml', 'dotenv', 'openai'
    ]
    
    print("\nChecking required modules:")
    for module in required_modules:
        try:
            __import__(module.replace('-', '_'))
            print(f"   ✓ {module}")
        except ImportError:
            print(f"   ✗ {module} - Not installed")
    
    # Check file structure
    required_files = [
        'config.yaml', '.env', 'requirements.txt', 
        'main.py', 'supervisor.py', 'worker.py', 'tools.py'
    ]
    
    print("\nChecking file structure:")
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file} - Missing")


async def main():
    """Main test function."""
    print("🚀 Grok Heavy System Test Suite")
    print("="*50)
    
    # Run diagnostics first
    run_diagnostics()
    
    print("\n" + "="*50)
    
    # Run basic functionality tests
    success = await test_basic_functionality()
    
    if success:
        # If basic tests pass, run tool tests
        await test_with_tools()
    
    print("\n" + "="*50)
    print("Test suite completed!")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\nTest suite failed with error: {e}")
        sys.exit(1) 