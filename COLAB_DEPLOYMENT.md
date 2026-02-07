# 🚀 Google Colab Deployment Guide

## Quick Start (5 minutes)

### Step 1: Upload Notebook to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `SurgeryPreview_Colab.ipynb` from this repository

### Step 2: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**

### Step 3: Run the Notebook

1. Click **Runtime** → **Run all**
2. Wait ~5 minutes for:
   - Dependencies to install
   - AI models to download (~650MB)
   - Application to launch
3. **Click the Gradio link** that appears (looks like: `https://xxxxx.gradio.live`)

### Step 4: Use the App

- The app will open in a new tab
- Share the link with colleagues or patients
- Upload expected result images and start previewing!

---

## 📊 What to Expect

| Metric | Value |
|--------|-------|
| **Setup Time** | ~5 minutes |
| **GPU** | T4 (Free) |
| **Performance** | 15-20 FPS |
| **Session Duration** | ~12 hours |
| **Cost** | FREE |

---

## 🔄 Alternative: Direct Colab Link

You can also create a direct Colab link by:

1. Uploading the notebook to your Google Drive
2. Right-click → **Open with** → **Google Colaboratory**
3. Share the Colab link with your team

---

## ⚠️ Important Limitations

### Free Tier Limits:
- **Session Duration**: ~12 hours, then disconnects
- **GPU Availability**: May switch to CPU after extended use
- **Idle Timeout**: Disconnects after 90 minutes of inactivity
- **Daily Limits**: Limited GPU hours per day

### Privacy Considerations:
- **Don't upload sensitive patient data** to public Colab
- For production use with real patient data, use a private deployment

---

## 🎯 Best Use Cases for Colab

✅ **Good for:**
- Testing and demos
- Showing the app to potential clients
- Training staff on how to use it
- Quick consultations (under 2 hours)

❌ **Not ideal for:**
- Production clinic use
- Storing patient data
- 24/7 availability
- Multiple concurrent users

---

## 💰 Upgrade Options

### Google Colab Pro ($10/month)
- Longer sessions (24 hours)
- Better GPUs (A100 available)
- No daily limits
- Priority access

### Google Colab Pro+ ($50/month)
- Background execution
- Even longer sessions
- More compute units

---

## 🚀 Next Steps for Production

If you need a production deployment:

1. **RunPod** - Best performance/price
   - RTX 4090: $0.44/hr
   - 60+ FPS
   - See `deploy/GPU_DEPLOYMENT_GUIDE.md`

2. **Vast.ai** - Cheapest option
   - RTX 3090: ~$0.30/hr
   - 45+ FPS

3. **Dedicated Server** - For hospitals/large clinics
   - Full control
   - HIPAA compliance possible
   - Contact for setup assistance

---

## 🆘 Troubleshooting

### "No GPU detected"
1. Runtime → Change runtime type → T4 GPU
2. Save and re-run all cells

### "Session disconnected"
1. Runtime → Restart runtime
2. Re-run all cells (models are cached, faster second time)

### "Out of memory"
1. Runtime → Restart runtime
2. Reduce image resolution in settings

### "Models not downloading"
1. Check internet connection
2. Try running the download cell again
3. Models are ~650MB total

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section in the notebook
- Review `README.md` for general app usage
- Check `deploy/GPU_DEPLOYMENT_GUIDE.md` for production options
