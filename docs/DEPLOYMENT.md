# 🏠 Local Deployment Guide

## Date: October 12, 2025
## Status: Ready for Local Deployment

---

## Quick Start (TL;DR)

```bash
cd "d:\ML Projects\FND"
python app_final.py
```

Then open your browser to: **http://localhost:8501**

---

## Prerequisites ✅

Your system already has everything installed:
- ✅ Python 3.13
- ✅ All required packages (TensorFlow, PyTorch, Transformers, etc.)
- ✅ All 3 models (SVM, LSTM, BERT) - verified working
- ✅ Compatibility layer active

---

## Step-by-Step Launch

### Option 1: Streamlit App (Recommended)

```powershell
# Navigate to project
cd "d:\ML Projects\FND"

# Launch Streamlit app
python -m streamlit run app_final.py

# Or simply:
streamlit run app_final.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Option 2: Custom Port

```powershell
streamlit run app_final.py --server.port 8080
```

### Option 3: Open to Network (Access from other devices)

```powershell
streamlit run app_final.py --server.address 0.0.0.0
```

---

## Features Available Locally

### ✅ Core Functionality
- **Real-time fake news detection**
- **Multi-model ensemble** (SVM + LSTM + BERT)
- **Credibility scoring** (0-100%)
- **Detailed analysis reports**

### ✅ Advanced Features
- **9 Credibility Analyzers:**
  - Bias Detection
  - Clickbait Detection
  - Emotional Manipulation
  - Linguistic Inconsistencies
  - Named Entity Recognition (NER)
  - Propaganda Detection
  - Sentiment Analysis
  - Source Credibility
  - Tone Analysis

### ✅ UI Features
- Clean, modern interface
- Real-time predictions
- Confidence scores
- Model-specific results
- Ensemble verdict

---

## Performance Expectations

### Startup Time
- **First Launch**: ~15 seconds (models loading)
- **Subsequent Launches**: ~10 seconds (cached)

### Prediction Speed
- **First Prediction**: ~2 seconds
- **Subsequent Predictions**: <1 second

### Resource Usage
- **RAM**: ~750MB-1GB
- **CPU**: 10-30% during prediction
- **Disk**: ~500MB for models

---

## Accessing the Application

### Local Access (Same Computer)
```
http://localhost:8501
```

### Network Access (Other Devices)
```
http://YOUR_IP_ADDRESS:8501
```

To find your IP:
```powershell
ipconfig
# Look for IPv4 Address under your active network adapter
```

---

## Troubleshooting

### Issue 1: Port Already in Use

**Error:**
```
Address already in use
```

**Solutions:**
```powershell
# Option A: Use different port
streamlit run app_final.py --server.port 8502

# Option B: Kill existing process
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Issue 2: Browser Doesn't Open

**Solution:**
Manually navigate to: `http://localhost:8501`

### Issue 3: Slow Loading

**Causes:**
- First-time BERT model download (~500MB)
- Antivirus scanning

**Solution:**
Wait for initial setup to complete. Subsequent runs will be faster.

### Issue 4: Model Loading Fails

**Check models exist:**
```powershell
dir models\
```

**Expected files:**
- `final_linear_svm.pkl`
- `final_vectorizer.pkl`
- `lstm_best_model.h5`
- `lstm_tokenizer.pkl`
- `bert_fake_news_model\` (folder)

**Re-verify:**
```powershell
python verify_environment.py
```

---

## Configuration Options

### Edit Streamlit Config

Create `.streamlit\config.toml`:

```toml
[server]
port = 8501
address = "localhost"
maxUploadSize = 200

[browser]
serverAddress = "localhost"
gatherUsageStats = false
serverPort = 8501

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Environment Variables

```powershell
# Disable TensorFlow warnings
$env:TF_CPP_MIN_LOG_LEVEL = "3"

# Set number of threads
$env:OMP_NUM_THREADS = "4"

# Then run app
streamlit run app_final.py
```

---

## Testing Your Local Deployment

### Test 1: Quick Verification

```powershell
python -c "from credibility_analyzer.credibility_analyzer import CredibilityAnalyzer; analyzer = CredibilityAnalyzer(); print(f'Models loaded: {len(analyzer.models)}/3')"
```

**Expected:** `Models loaded: 3/3`

### Test 2: Sample Prediction

```powershell
python test_prediction.py
```

### Test 3: Full Test Suite

```powershell
python test_compatibility.py
```

---

## Usage Examples

### Example 1: Political News
```
Input: "Breaking: Local mayor announces new infrastructure project with bipartisan support"
Expected: High credibility (70-90%)
```

### Example 2: Sensational Headline
```
Input: "You won't BELIEVE what this celebrity did! Doctors HATE this one trick!"
Expected: Low credibility (10-30%)
```

### Example 3: Factual Report
```
Input: "The Federal Reserve announced a 0.25% interest rate increase in today's meeting"
Expected: High credibility (80-95%)
```

---

## Stopping the Application

### Method 1: Terminal
Press `Ctrl + C` in the terminal

### Method 2: Close Browser
Just close the browser tab (app keeps running)

### Method 3: Kill Process
```powershell
# Find Streamlit process
Get-Process | Where-Object {$_.ProcessName -like "*python*"}

# Kill specific process
Stop-Process -Name python -Force
```

---

## Auto-Start on System Boot (Optional)

### Create Startup Script

**File:** `start_fake_news_detector.bat`

```batch
@echo off
cd /d "d:\ML Projects\FND"
start /min streamlit run app_final.py
```

**Add to Startup:**
1. Press `Win + R`
2. Type: `shell:startup`
3. Copy the `.bat` file there

---

## Sharing Access on Local Network

### Step 1: Find Your IP Address

```powershell
ipconfig
```

Look for: `IPv4 Address: 192.168.x.x`

### Step 2: Start with Network Access

```powershell
streamlit run app_final.py --server.address 0.0.0.0
```

### Step 3: Share URL

Give others: `http://192.168.x.x:8501`

### Step 4: Configure Firewall (if needed)

```powershell
# Allow Streamlit through firewall
New-NetFirewallRule -DisplayName "Streamlit App" -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow
```

---

## Performance Optimization

### For Faster Startup

**Option 1: Lazy Loading**
Only load models when first prediction is requested

**Option 2: Pre-warm Models**
Create a warmup prediction on startup

### For Lower Memory Usage

**Option 1: Run Single Model**
Modify `credibility_analyzer.py` to load only BERT (most accurate)

**Option 2: Clear Cache**
```python
import gc
gc.collect()
```

---

## Development Mode

### Enable Auto-Reload

```powershell
streamlit run app_final.py --server.runOnSave true
```

Changes to code will automatically refresh the app.

### Enable Debug Mode

```powershell
streamlit run app_final.py --logger.level=debug
```

---

## Backup & Recovery

### Backup Your Setup

```powershell
# Backup models
Copy-Item -Path "models" -Destination "models_backup" -Recurse

# Backup configuration
Copy-Item -Path "*.py" -Destination "backup\" -Recurse
```

### Quick Recovery

If something breaks:
```powershell
# Restore models
Remove-Item -Path "models" -Recurse -Force
Copy-Item -Path "models_backup" -Destination "models" -Recurse

# Re-verify
python verify_environment.py
```

---

## Monitoring & Logs

### View Streamlit Logs

Logs are saved to: `%USERPROFILE%\.streamlit\logs\`

### View Application Logs

```powershell
# Real-time monitoring
Get-Content -Path "app.log" -Wait
```

### Monitor Resource Usage

```powershell
# Check memory usage
Get-Process python | Select-Object Name, CPU, WorkingSet

# Check CPU usage
Get-Counter '\Processor(_Total)\% Processor Time'
```

---

## Security Considerations

### For Local Use Only
✅ Default settings are secure
✅ Only accessible from localhost
✅ No external exposure

### For Network Sharing
⚠️ Consider adding authentication
⚠️ Use HTTPS if handling sensitive data
⚠️ Restrict firewall rules to trusted IPs

---

## Advantages of Local Deployment

### ✅ Benefits
- **Full Control**: No platform restrictions
- **Privacy**: Data stays on your machine
- **No Costs**: Free forever
- **Fast**: No network latency
- **Customizable**: Modify freely
- **Offline**: Works without internet (except first BERT download)

### ⚠️ Considerations
- Single-user by default
- Requires your computer running
- You manage updates
- No automatic scaling

---

## Next Steps

1. **Launch the app**: `streamlit run app_final.py`
2. **Test with sample news**: Verify predictions work
3. **Customize if needed**: Modify UI, add features
4. **Share locally**: Give network access to teammates
5. **Monitor performance**: Check logs and metrics

---

## Quick Commands Reference

```powershell
# Start app
streamlit run app_final.py

# Start on custom port
streamlit run app_final.py --server.port 8080

# Start with network access
streamlit run app_final.py --server.address 0.0.0.0

# Stop app
Ctrl + C

# Verify models
python verify_environment.py

# Run tests
python test_compatibility.py

# Check app status
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"}
```

---

## Support

### Documentation
- **This Guide**: `LOCAL_DEPLOYMENT_GUIDE.md`
- **Compatibility Info**: `COMPATIBILITY_FIXES_SUMMARY.md`
- **Test Results**: `test_compatibility.py`

### Troubleshooting
- Check terminal output for errors
- Review `verify_environment.py` results
- Test individual models: `python test_prediction.py`

---

**Status**: ✅ Ready for Local Use  
**Last Updated**: October 12, 2025  
**Tested**: Python 3.13, Windows 11  

🎉 **Enjoy your local fake news detection system!**
