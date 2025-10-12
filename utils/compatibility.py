"""
Compatibility utilities for handling version conflicts in Hugging Face Spaces
Handles numpy, TensorFlow, and PyTorch version compatibility issues
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def fix_numpy_compatibility():
    """Fix numpy compatibility issues with older pickle files."""
    try:
        import numpy as np
        # Create numpy._core alias for backward compatibility
        if not hasattr(np, '_core'):
            import numpy.core as _core
            np._core = _core
            sys.modules['numpy._core'] = _core
            sys.modules['numpy._core.multiarray'] = _core.multiarray
            sys.modules['numpy._core.umath'] = _core.umath
        return True
    except Exception as e:
        print(f"Warning: Could not fix numpy compatibility: {e}")
        return False

def safe_load_pickle(filepath):
    """Safely load pickle files with numpy compatibility."""
    import pickle
    import numpy as np
    
    # Ensure numpy compatibility is fixed first
    fix_numpy_compatibility()
    
    try:
        # Try standard loading first
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError) as e:
        if 'numpy._core' in str(e) or 'numpy.core' in str(e):
            # Apply fix and retry
            fix_numpy_compatibility()
            with open(filepath, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        else:
            raise
    except Exception:
        # Try with different encoding as last resort
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except:
            # Final attempt with bytes encoding
            with open(filepath, 'rb') as f:
                return pickle.load(f, encoding='bytes')

def safe_load_keras_model(filepath):
    """Safely load Keras models with TensorFlow compatibility."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Custom objects for compatibility - handle common issues
        custom_objects = {
            # Add any custom layers/objects here if needed
        }
        
        try:
            # Try loading without compilation first (handles batch_shape issue)
            model = keras.models.load_model(
                filepath, 
                custom_objects=custom_objects, 
                compile=False
            )
        except Exception as e:
            if 'batch_shape' in str(e):
                # Specific handling for batch_shape issue
                # This is a known issue with older Keras models
                print(f"⚠️ Handling batch_shape compatibility issue...")
                
                # Try with safe mode
                import h5py
                with h5py.File(filepath, 'r') as f:
                    # Load architecture
                    model_config = f.attrs.get('model_config')
                    if model_config is None:
                        model_config = f.attrs.get('model_config')
                    
                    # Reconstruct model from config (bypass batch_shape)
                    from tensorflow.keras.models import model_from_json
                    if isinstance(model_config, bytes):
                        model_config = model_config.decode('utf-8')
                    
                    # Remove batch_shape from config
                    import json
                    config = json.loads(model_config)
                    
                    # Clean config recursively
                    def clean_config(obj):
                        if isinstance(obj, dict):
                            obj.pop('batch_shape', None)
                            if 'config' in obj and isinstance(obj['config'], dict):
                                obj['config'].pop('batch_shape', None)
                                # Replace with input_shape if available
                                if 'batch_input_shape' in obj['config']:
                                    shape = obj['config']['batch_input_shape']
                                    if shape and len(shape) > 1:
                                        obj['config']['input_shape'] = shape[1:]
                            for v in obj.values():
                                clean_config(v)
                        elif isinstance(obj, list):
                            for item in obj:
                                clean_config(item)
                    
                    clean_config(config)
                    model = model_from_json(json.dumps(config))
                    
                    # Load weights
                    model.load_weights(filepath)
            else:
                raise
        
        # Compile manually with safe defaults
        try:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as compile_error:
            print(f"⚠️ Could not compile model: {compile_error}")
            # Model can still be used for inference without compilation
        
        return model
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        raise

def safe_load_transformers_model(model_name, local_path=None):
    """Safely load transformers models without accelerate dependency."""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        load_kwargs = {
            'low_cpu_mem_usage': False,  # Disable to avoid accelerate
            'torch_dtype': None,  # Use default dtype
        }
        
        if local_path and os.path.exists(local_path):
            model = AutoModel.from_pretrained(local_path, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(local_path)
        else:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading transformers model: {e}")
        raise

def check_dependencies():
    """Check and report dependency versions."""
    deps = {}
    
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = 'not installed'
    
    try:
        import tensorflow
        deps['tensorflow'] = tensorflow.__version__
    except ImportError:
        deps['tensorflow'] = 'not installed'
    
    try:
        import torch
        deps['torch'] = torch.__version__
    except ImportError:
        deps['torch'] = 'not installed'
    
    try:
        import transformers
        deps['transformers'] = transformers.__version__
    except ImportError:
        deps['transformers'] = 'not installed'
    
    try:
        import sklearn
        deps['scikit-learn'] = sklearn.__version__
    except ImportError:
        deps['scikit-learn'] = 'not installed'
    
    return deps

def print_dependency_report():
    """Print a formatted dependency report."""
    print("\n" + "="*50)
    print("DEPENDENCY VERSION REPORT")
    print("="*50)
    
    deps = check_dependencies()
    for name, version in deps.items():
        status = "✅" if version != 'not installed' else "❌"
        print(f"{status} {name}: {version}")
    
    print("="*50 + "\n")

# Auto-fix numpy compatibility on import
fix_numpy_compatibility()
