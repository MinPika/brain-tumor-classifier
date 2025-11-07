# 🧠 Brain Tumor MRI Classification - Streamlit App

## 📋 Features

- **Single Image Analysis:** Upload and classify individual MRI scans
- **Batch Processing:** Analyze multiple images at once
- **Interactive Visualizations:** Confidence scores with bar charts and pie charts
- **Detailed Reports:** Download classification reports as TXT files
- **Batch CSV Export:** Export batch analysis results
- **Responsive Design:** Clean white background, modern UI
- **Educational Info:** Learn about tumor types and treatments

## 🚀 Deployment Steps to Streamlit Cloud

### Step 1: Prepare Your Files

Ensure you have these files in your project directory:
```
brain-tumor-app/
├── streamlit_app.py          # Main application file
├── requirements.txt           # Python dependencies
├── model.keras               # Trained model (REQUIRED)
└── README.md                 # This file
```

### Step 2: Upload Model to GitHub

1. **Create a GitHub repository:**
   ```bash
   git init
   git add streamlit_app.py requirements.txt README.md
   git commit -m "Initial commit"
   ```

2. **For model.keras file (2 options):**

   **Option A: GitHub (if < 100MB)**
   ```bash
   git add model.keras
   git commit -m "Add model"
   git push origin main
   ```

   **Option B: GitHub Large File Storage (if > 100MB)**
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes model.keras
   git commit -m "Add model with LFS"
   git push origin main
   ```

   **Option C: External Storage (Recommended)**
   - Upload `model.keras` to Google Drive/Dropbox
   - Get direct download link
   - Modify `streamlit_app.py` to download on first run:
   
   ```python
   import gdown
   
   @st.cache_resource
   def load_model():
       if not os.path.exists('model.keras'):
           url = 'YOUR_GOOGLE_DRIVE_LINK'
           gdown.download(url, 'model.keras', quiet=False)
       return tf.keras.models.load_model('model.keras')
   ```

### Step 3: Deploy to Streamlit Cloud

1. **Go to:** https://streamlit.io/cloud

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Fill in deployment settings:**
   - Repository: `your-username/brain-tumor-app`
   - Branch: `main`
   - Main file path: `streamlit_app.py`

5. **Advanced settings (Optional):**
   - Python version: 3.11
   - Add secrets if using external APIs

6. **Click "Deploy"**

7. **Wait 2-5 minutes** for deployment

8. **Your app will be live at:** `https://your-app-name.streamlit.app`

### Step 4: Test Your Deployment

1. Visit your app URL
2. Upload a test MRI image
3. Verify predictions work correctly
4. Test batch processing
5. Check all pages load properly

## 🔧 Local Testing (Before Deployment)

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

Access at: http://localhost:8501

## 📦 Alternative Deployment Options

### Option 1: Hugging Face Spaces

1. Create account at https://huggingface.co
2. Create new Space
3. Upload files
4. Select "Streamlit" as SDK
5. Deploy

### Option 2: Railway

1. Sign up at https://railway.app
2. Create new project
3. Connect GitHub repo
4. Add start command: `streamlit run streamlit_app.py --server.port=$PORT`
5. Deploy

### Option 3: Google Cloud Run

```bash
# Create Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]

# Deploy
gcloud run deploy brain-tumor-app --source .
```

## ⚙️ Configuration Options

### Customize App Theme

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#0EEAF1"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Add Custom Domain

In Streamlit Cloud:
1. Go to App Settings
2. Click "Custom Domain"
3. Follow DNS configuration steps

## 🐛 Troubleshooting

### Model File Too Large
- Use Git LFS
- Or use external storage (Google Drive + gdown)

### App Crashes on Startup
- Check `requirements.txt` versions
- Verify `model.keras` is accessible
- Check Streamlit Cloud logs

### Slow Performance
- Reduce image size in preprocessing
- Enable caching with `@st.cache_resource`
- Use Streamlit Cloud's higher tier

### Memory Issues
- Streamlit free tier: 1GB RAM
- Optimize model loading
- Clear cache periodically

## 📊 App Structure

```
streamlit_app.py
├── Page 1: Home & Upload
│   ├── Single image upload
│   ├── Prediction display
│   ├── Confidence visualization
│   └── Download report
├── Page 2: Batch Analysis
│   ├── Multiple image upload
│   ├── Progress tracking
│   ├── Summary statistics
│   └── CSV export
├── Page 3: About
│   ├── Model information
│   ├── Performance metrics
│   └── Architecture details
└── Page 4: Help
    ├── User guide
    ├── Best practices
    └── Troubleshooting
```

## 🔐 Security Notes

- App includes medical disclaimer
- No data is stored permanently
- All processing happens in-session
- Images are not logged or saved

## 📈 Performance Metrics

- **Model Accuracy:** 99.31%
- **Response Time:** ~1-2 seconds per image
- **Supported Formats:** JPG, JPEG, PNG
- **Max Upload Size:** 200MB (Streamlit default)

## 🤝 Support

For issues or questions:
1. Check logs in Streamlit Cloud dashboard
2. Review GitHub Issues
3. Contact: your-email@example.com

## 📄 License

MIT License - Feel free to modify and distribute

## 🙏 Acknowledgments

- Dataset: Figshare, Br35H, SARTAJ
- Framework: TensorFlow 2.16
- Deployment: Streamlit Cloud
- Course: ESL372 - Neural Networks

---

**⚠️ Important:** This application is for educational purposes only. Not intended for clinical diagnosis. Always consult medical professionals.