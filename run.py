#!/usr/bin/env python3
"""
Quick startup script for Grok Heavy Multi-Agent AI System.
Provides an easy way to launch the system with common configurations.
"""

import os
import sys
import subprocess


def check_dependencies():
    """Check if all dependencies are installed."""
    try:
        import aiohttp
        import yaml
        import dotenv
        import openai
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def check_environment():
    """Check if environment is properly configured."""
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("Please create a .env file with your OPENROUTER_API_KEY and EXA_API_KEY")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        print("Please add your OpenRouter API key to the .env file")
        return False
    
    if not os.getenv('EXA_API_KEY'):
        print("⚠️  EXA_API_KEY not found in .env file!")
        print("Web search functionality will be limited without Exa API key")
    
    return True


def main():
    """Main startup function."""
    print("🚀 Grok Heavy - Multi-Agent AI System")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies OK")
    
    # Check environment
    print("Checking environment...")
    if not check_environment():
        sys.exit(1)
    print("✅ Environment OK")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Pass arguments to main.py
        cmd = [sys.executable, 'main.py'] + sys.argv[1:]
        subprocess.run(cmd)
    else:
        # Show menu
        print("\nChoose an option:")
        print("1. Interactive Mode (recommended for first-time users)")
        print("2. Run Test Suite")
        print("3. Quick Query")
        print("4. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\nStarting Interactive Mode...")
                subprocess.run([sys.executable, 'main.py'])
                break
            elif choice == '2':
                print("\nRunning Test Suite...")
                subprocess.run([sys.executable, 'test_system.py'])
                break
            elif choice == '3':
                query = input("Enter your query: ").strip()
                if query:
                    subprocess.run([sys.executable, 'main.py', query])
                break
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == '__main__':
    main() 