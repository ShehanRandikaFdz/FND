"""
Environment Verification Script for Hugging Face Spaces
Run this before starting the Streamlit app to verify all dependencies
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_environment():
    """Comprehensive environment check."""
    print("\n" + "="*70)
    print("🔍 ENVIRONMENT VERIFICATION - Fake News Detection System")
    print("="*70 + "\n")
    
    issues = []
    warnings = []
    
    # 1. Check Python version
    print("1️⃣ Checking Python version...")
    python_version = sys.version_info
    print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        issues.append("Python version too old (requires 3.8+)")
    print("   ✅ Python version OK\n")
    
    # 2. Check critical dependencies
    print("2️⃣ Checking dependencies...")
    
    deps_to_check = {
        'numpy': '1.23.5',
        'tensorflow': '2.13.1',
        'torch': '2.0.1',
        'transformers': '4.33.2',
        'sklearn': '1.3.0',
        'streamlit': '1.28.1',
        'pandas': '2.0.3',
        'joblib': '1.3.2',
    }
    
    for module_name, expected_version in deps_to_check.items():
        try:
            if module_name == 'sklearn':
                import sklearn
                module = sklearn
            else:
                module = __import__(module_name)
            
            actual_version = module.__version__
            status = "✅" if actual_version.startswith(expected_version.split('.')[0]) else "⚠️"
            print(f"   {status} {module_name}: {actual_version} (expected: {expected_version})")
            
            if not actual_version.startswith(expected_version.split('.')[0]):
                warnings.append(f"{module_name} version mismatch: {actual_version} vs {expected_version}")
        except ImportError:
            print(f"   ❌ {module_name}: NOT INSTALLED")
            issues.append(f"{module_name} not installed")
        except AttributeError:
            print(f"   ⚠️ {module_name}: Installed but version unclear")
    
    print()
    
    # 3. Check numpy compatibility
    print("3️⃣ Checking numpy compatibility...")
    try:
        import numpy as np
        if hasattr(np, '_core'):
            print("   ✅ numpy._core available")
        else:
            print("   ⚠️ numpy._core not available, applying compatibility fix...")
            from utils.compatibility import fix_numpy_compatibility
            if fix_numpy_compatibility():
                print("   ✅ Compatibility fix applied")
            else:
                warnings.append("numpy compatibility fix failed")
                print("   ⚠️ Compatibility fix failed")
    except Exception as e:
        issues.append(f"numpy compatibility check failed: {e}")
        print(f"   ❌ Error: {e}")
    print()
    
    # 4. Check model files
    print("4️⃣ Checking model files...")
    models_dir = "models"
    
    required_files = [
        "new_svm_model.pkl",
        "new_svm_vectorizer.pkl",
        "lstm_fake_news_model.h5",
        "lstm_tokenizer.pkl",
        "bert_fake_news_model/classifier.pkl",
    ]
    
    for file_path in required_files:
        full_path = os.path.join(models_dir, file_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.2f} MB)")
        else:
            warnings.append(f"Model file not found: {file_path}")
            print(f"   ⚠️ {file_path} - NOT FOUND")
    print()
    
    # 5. Test model loading
    print("5️⃣ Testing model loading...")
    try:
        from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer
        print("   Testing CredibilityAnalyzer initialization...")
        analyzer = CredibilityAnalyzer(models_dir=models_dir)
        
        models_loaded = list(analyzer.models.keys())
        print(f"   ✅ Models loaded: {', '.join(models_loaded) if models_loaded else 'None'}")
        
        if not models_loaded:
            issues.append("No models could be loaded")
        elif len(models_loaded) < 3:
            warnings.append(f"Only {len(models_loaded)}/3 models loaded")
    except Exception as e:
        issues.append(f"Model loading test failed: {e}")
        print(f"   ❌ Error: {e}")
    print()
    
    # 6. Memory check
    print("6️⃣ Checking system resources...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   Total Memory: {memory.total / (1024**3):.2f} GB")
        print(f"   Available Memory: {memory.available / (1024**3):.2f} GB")
        print(f"   Memory Usage: {memory.percent}%")
        
        if memory.available < 1024**3:  # Less than 1GB
            warnings.append("Low memory available (< 1GB)")
            print("   ⚠️ Low memory available")
        else:
            print("   ✅ Sufficient memory")
    except ImportError:
        print("   ⚠️ psutil not available, skipping memory check")
    print()
    
    # Summary
    print("="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    if not issues and not warnings:
        print("✅ All checks passed! System ready for deployment.")
        return 0
    elif issues:
        print(f"❌ {len(issues)} critical issue(s) found:")
        for issue in issues:
            print(f"   • {issue}")
        if warnings:
            print(f"\n⚠️ {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"   • {warning}")
        print("\n❌ System NOT ready for deployment. Please fix critical issues.")
        return 1
    else:
        print(f"⚠️ {len(warnings)} warning(s) found:")
        for warning in warnings:
            print(f"   • {warning}")
        print("\n⚠️ System may work with degraded functionality.")
        return 0
    
    print("="*70 + "\n")

if __name__ == "__main__":
    exit_code = check_environment()
    sys.exit(exit_code)
