#!/usr/bin/env python3
"""
Setup script for Grok Heavy Multi-Agent AI System.
Handles installation, dependency management, and initial configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def setup_environment():
    """Setup environment file if it doesn't exist."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    print("🔧 Setting up environment file...")
    
    openrouter_key = input("Enter your OpenRouter API key: ").strip()
    
    if not openrouter_key:
        print("❌ OpenRouter API key is required")
        return False
    
    exa_key = input("Enter your Exa API key (optional, press Enter to skip): ").strip()
    
    try:
        with open(".env", "w") as f:
            f.write(f"OPENROUTER_API_KEY={openrouter_key}\n")
            if exa_key:
                f.write(f"EXA_API_KEY={exa_key}\n")
        
        print("✅ Environment file created")
        if not exa_key:
            print("⚠️  Note: Web search functionality will be limited without Exa API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create environment file: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = ["logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def run_test():
    """Run system test to verify installation."""
    print("🧪 Running system test...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_system.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ System test passed")
            return True
        else:
            print("❌ System test failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


def display_usage_info():
    """Display usage information."""
    print("\n" + "="*60)
    print("🎉 GROK HEAVY SETUP COMPLETE!")
    print("="*60)
    print("\nQuick Start:")
    print("  python run.py                    # Launch with menu")
    print("  python main.py                   # Interactive mode")
    print('  python main.py "your question"   # Direct query')
    print("  python main.py --help            # Show all options")
    print("\nExamples:")
    print('  python main.py "What is quantum computing?"')
    print('  python main.py "Write Python code to sort a list" --verbose')
    print("\nFiles created:")
    print("  📁 logs/          # Session logs will be saved here")
    print("  📄 .env           # Your API key configuration")
    print("  📄 grok_heavy.log # System log file")
    print("\nFor more information, see README.md")
    print("="*60)


def main():
    """Main setup function."""
    print("🚀 Grok Heavy Multi-Agent AI System Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run test
    print("\nWould you like to run a system test? (y/n): ", end="")
    if input().strip().lower() in ['y', 'yes']:
        if not run_test():
            print("⚠️  Warning: System test failed, but setup is complete")
            print("   You can try running 'python test_system.py' manually")
    
    # Display usage info
    display_usage_info()


def clean():
    """Clean up generated files and directories."""
    print("🧹 Cleaning up Grok Heavy installation...")
    
    items_to_remove = [
        "logs/",
        "__pycache__/",
        "*.log",
        ".env"
    ]
    
    for item in items_to_remove:
        if "*" in item:
            # Handle glob patterns
            import glob
            for file in glob.glob(item):
                try:
                    os.remove(file)
                    print(f"   Removed: {file}")
                except OSError:
                    pass
        else:
            path = Path(item)
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    print(f"   Removed directory: {path}")
                except OSError:
                    pass
            elif path.is_file():
                try:
                    path.unlink()
                    print(f"   Removed file: {path}")
                except OSError:
                    pass
    
    print("✅ Cleanup complete")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'clean':
        clean()
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\n\nSetup interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1) 