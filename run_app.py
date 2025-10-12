#!/usr/bin/env python3
"""
Simple launcher for the Unified News Verification System
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app"""
    print("🚀 Starting Unified News Verification System...")
    print("📝 This will open in your browser at http://localhost:8501")
    print("⏳ Please wait while the system loads...")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the FND directory")
        print("2. Try: python -m streamlit run app.py")
        print("3. Check if all dependencies are installed")

if __name__ == "__main__":
    main()
