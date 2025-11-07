"""
Pre-Deployment Test Script
Run this before deploying to catch issues early
"""

import os
import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing imports...")
    
    required = [
        ('streamlit', 'streamlit'),
        ('tensorflow', 'tensorflow'),
        ('numpy', 'np'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL'),
        ('plotly', 'plotly'),
        ('pandas', 'pd'),
        ('sklearn', 'sklearn')
    ]
    
    failed = []
    for package, alias in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(failed)}")
        return False
    
    print("✅ All imports successful!\n")
    return True

def test_files():
    """Check if all required files exist"""
    print("🔍 Testing file structure...")
    
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        'model.keras',
        'README.md'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   ✅ {file} ({size:.2f} MB)")
        else:
            print(f"   ❌ {file} - NOT FOUND")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        if 'model.keras' in missing:
            print("   💡 Download model from Kaggle or use Google Drive method")
        return False
    
    print("✅ All required files present!\n")
    return True

def test_model():
    """Test if model can be loaded"""
    print("🔍 Testing model loading...")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model.keras')
        print(f"   ✅ Model loaded successfully")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
        print(f"   📊 Parameters: {model.count_params():,}")
        print("✅ Model test passed!\n")
        return True
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

def test_requirements():
    """Check requirements.txt format"""
    print("🔍 Testing requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            lines = f.readlines()
        
        print(f"   📦 Found {len(lines)} packages")
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                package = line.split('==')[0]
                print(f"   - {line}")
        
        print("✅ Requirements file valid!\n")
        return True
    except Exception as e:
        print(f"   ❌ Error reading requirements.txt: {e}")
        return False

def test_app_syntax():
    """Check if streamlit_app.py has syntax errors"""
    print("🔍 Testing app syntax...")
    
    try:
        with open('streamlit_app.py', 'r') as f:
            code = f.read()
        
        compile(code, 'streamlit_app.py', 'exec')
        print("   ✅ No syntax errors")
        print("✅ App syntax valid!\n")
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False

def test_model_prediction():
    """Test if model can make predictions"""
    print("🔍 Testing model prediction...")
    
    try:
        import tensorflow as tf
        import numpy as np
        
        model = tf.keras.models.load_model('model.keras')
        
        # Create dummy input
        dummy_input = np.random.rand(1, 168, 168, 1)
        
        # Test prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"   ✅ Prediction shape: {prediction.shape}")
        print(f"   ✅ Prediction sum: {prediction.sum():.4f} (should be ~1.0)")
        print(f"   ✅ Max probability: {prediction.max()*100:.2f}%")
        print("✅ Prediction test passed!\n")
        return True
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")
        return False

def estimate_memory():
    """Estimate memory usage"""
    print("🔍 Estimating memory usage...")
    
    try:
        import tensorflow as tf
        
        model = tf.keras.models.load_model('model.keras')
        
        # Get model size
        model_size = os.path.getsize('model.keras') / (1024*1024)
        
        # Estimate runtime memory
        params = model.count_params()
        estimated_memory = (params * 4) / (1024*1024)  # 4 bytes per param
        
        print(f"   📊 Model file: {model_size:.2f} MB")
        print(f"   📊 Estimated RAM: {estimated_memory:.2f} MB")
        print(f"   📊 Streamlit free tier: 1000 MB")
        
        if model_size < 100:
            print("   ✅ Can use GitHub directly")
        else:
            print("   ⚠️  Use Git LFS or Google Drive")
        
        if estimated_memory < 500:
            print("   ✅ Memory usage OK for Streamlit")
        else:
            print("   ⚠️  Consider model optimization")
        
        print("✅ Memory estimation complete!\n")
        return True
    except Exception as e:
        print(f"   ❌ Memory estimation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("🚀 PRE-DEPLOYMENT TEST SUITE")
    print("="*60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("File Structure", test_files),
        ("Requirements", test_requirements),
        ("App Syntax", test_app_syntax),
        ("Model Loading", test_model),
        ("Model Prediction", test_model_prediction),
        ("Memory Estimation", estimate_memory)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}\n")
            results.append((name, False))
    
    # Summary
    print("="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12s} - {name}")
    
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready to deploy!")
        print("\nNext steps:")
        print("1. Push to GitHub: git push origin main")
        print("2. Go to: https://share.streamlit.io")
        print("3. Click 'New app' and deploy!")
    else:
        print("⚠️  Fix failed tests before deploying")
        print("\nCommon fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Download model.keras from Kaggle")
        print("- Check syntax errors in streamlit_app.py")
    
    print("="*60)

if __name__ == "__main__":
    run_all_tests()