# 🌱 Plant Piano Web App - Quick Start Guide

## 🎯 What You Just Got

A beautiful, minimal web interface for your Plant Piano with:
- ✅ **2 Modes**: Upload recordings OR live detection
- ✅ **3 Models**: Switch between Random Forest, Gradient Boosting, Logistic Regression
- ✅ **Threshold Slider**: Adjust sensitivity (0.5 - 0.95)
- ✅ **11 Musical Notes**: Extended scale C3-A3-C4-A4-C5
- ✅ **Visual Piano Keyboard**: See notes light up as they play
- ✅ **Waveform Visualization**: See your plant's signal
- ✅ **Beautiful Design**: Plant & music themed with gradients
- ✅ **Future-Ready**: Space for adding new instruments

---

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements_web.txt
```

If you get errors, try:
```bash
pip install flask flask-socketio flask-cors python-socketio numpy scipy scikit-learn joblib eventlet
```

### Step 2: Verify File Structure

Make sure you have:
```
plant-piano/
├── app.py                          # Backend server
├── templates/
│   └── index.html                  # Frontend interface
├── output/
│   └── models/
│       ├── random_forest_model.pkl
│       ├── gradient_boost_model.pkl
│       └── logistic_model.pkl
└── Dataset/
    └── Holy_Basil/
        └── (your WAV files)
```

### Step 3: Start the Server

```bash
python app.py
```

You should see:
```
╔══════════════════════════════════════════════════════════════════════════╗
║                    🌱 PLANT PIANO WEB SERVER 🎹                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Starting server...
Open your browser to: http://localhost:5000
```

### Step 4: Open Your Browser

Navigate to: **http://localhost:5000**

You'll see a beautiful purple gradient interface with plant and music emojis! 🌱🎹

---

## 🎹 How to Use

### **Upload Mode** (Test on Recordings)

1. **Select Upload Mode** (already selected by default)

2. **Choose Model** (Random Forest recommended)

3. **Adjust Threshold**:
   - **0.7** = Balanced (recommended)
   - **0.5-0.6** = Sensitive (catches light touches, might get noise)
   - **0.8-0.9** = Strict (only obvious touches)

4. **Upload WAV File**:
   - Click the upload area or drag & drop
   - Use your `touch_02.wav`, `touch_11.wav`, etc.

5. **Click "Process & Play Music"**

6. **Watch & Listen**! 🎵
   - Piano keys light up
   - Notes play automatically
   - See waveform with detection markers
   - Results list shows all detected touches

### **Live Mode** (Real-Time with Plant SpikerBox)

**Note**: Live mode needs audio routing setup (see below)

1. **Click "Live Mode" card**

2. **Setup**:
   - Open BYB Spike Recorder
   - Enable "Audio Output" in settings
   - Use Virtual Audio Cable or loopback

3. **Click "Start Listening"**

4. **Touch your plant** - hear notes in real-time!

---

## 🎵 Musical Note Mapping

Your touches now play **11 different notes** based on amplitude:

| Amplitude Range | Note | Frequency | Touch Type |
|----------------|------|-----------|------------|
| < 0.004 | C3 | 131 Hz | Very gentle whisper |
| 0.004-0.008 | D3 | 147 Hz | Gentle |
| 0.008-0.012 | E3 | 165 Hz | Light touch |
| 0.012-0.020 | G3 | 196 Hz | Medium-light |
| 0.020-0.028 | A3 | 220 Hz | Medium |
| 0.028-0.035 | C4 | 262 Hz | Medium-firm |
| 0.035-0.045 | D4 | 294 Hz | Firm |
| 0.045-0.055 | E4 | 330 Hz | Strong |
| 0.055-0.070 | G4 | 392 Hz | Very strong |
| 0.070-0.090 | A4 | 440 Hz | Hard |
| > 0.090 | C5 | 523 Hz | Very hard |

**Your recordings**:
- Gentle touches → C3, D3, E3, G3
- Medium touches → A3, C4, D4, E4
- Strong touches → G4, A4, C5

More variety = More musical expression! 🎶

---

## 🎨 Interface Features

### **Header**
- Beautiful gradient background (purple → violet)
- Plant emojis (🌿) decorating the corners
- Clear title and subtitle

### **Mode Cards**
- **Upload** 📁: Test on recorded files
- **Live** 🎤: Real-time detection
- Cards glow green when selected

### **Controls Panel**
- **Model Selector**: Dropdown with 3 trained models
- **Threshold Slider**: Real-time adjustment with visual value
- Helpful hints below slider

### **Upload Area**
- Drag & drop support
- Click to browse
- Visual feedback (hover, dragover effects)

### **Visualization**
- **Signal Waveform**: Canvas-based line graph
- **Green markers** at detection points
- **Note labels** above markers

### **Piano Keyboard**
- **11 keys** (C3 to C5)
- Keys light up **green** when playing
- Click keys to test sounds manually
- Smooth animations

### **Results List**
- Time of each detection
- Note played
- Confidence percentage (color-coded badge)
- Scrollable list

---

## 🎨 Color Scheme

- **Primary Green**: `#2ECC71` - Plant theme
- **Secondary Green**: `#27AE60` - Darker shade
- **Accent Orange**: `#F39C12` - Highlights
- **Purple Gradient**: Background
- **White/Light Gray**: Cards and content

---

## 🔧 Customization Ideas

### **Add More Instruments** (Future)

In `app.py`, add after line 45:
```python
INSTRUMENTS = {
    'piano': {...},  # Current notes
    'marimba': {...},  # Different timbre
    'flute': {...},   # Softer sound
    'guitar': {...}   # Plucked sound
}
```

In `index.html`, add instrument selector:
```html
<select id="instrument-select">
    <option value="piano">🎹 Piano</option>
    <option value="marimba">🎵 Marimba</option>
    <option value="flute">🎺 Flute</option>
</select>
```

### **Change Color Theme**

Edit `index.html` CSS variables (around line 17):
```css
:root {
    --primary: #2ECC71;    /* Change to your color */
    --secondary: #27AE60;  /* Darker shade */
    --accent: #F39C12;     /* Accent color */
}
```

Try:
- **Blue theme**: `#3498DB`, `#2980B9`, `#E74C3C`
- **Pink theme**: `#E91E63`, `#C2185B`, `#FFC107`
- **Forest theme**: `#27AE60`, `#1E8449`, `#F39C12`

### **Adjust Note Ranges**

In `app.py`, function `amplitude_to_note()` (around line 70):
```python
def amplitude_to_note(amplitude):
    if amplitude < 0.004:
        return 'C3'
    elif amplitude < 0.010:  # Changed from 0.008
        return 'D3'
    # ... etc
```

---

## 🐛 Troubleshooting

### **"Model not found" error**
- Make sure you ran `train_model.py` first
- Check `output/models/` folder exists with .pkl files

### **No audio playing**
- Check browser console (F12) for errors
- Try clicking anywhere on page first (browser autoplay policy)
- Make sure volume is up!

### **Upload button stays disabled**
- Only WAV files accepted
- File must be selected/dropped first

### **Live mode doesn't work**
- Requires audio routing setup
- Use Virtual Audio Cable (Windows) or BlackHole (Mac)
- Or use Upload mode with recorded files

### **Notes don't match expected**
- Check threshold setting
- Try different models
- Your plant's response might vary!

### **Server won't start**
- Port 5000 might be in use: `lsof -i :5000` (Mac/Linux)
- Or change port in `app.py` last line: `port=5001`

---

## 📊 Performance Tips

### **Faster Processing**
- Use **Random Forest** (fastest)
- Lower sample rate in recordings
- Process shorter files first

### **Better Detection**
- Use **threshold 0.7-0.8** for clean signals
- **Gradient Boosting** often most accurate
- Record in quiet environment

### **More Musical**
- Record varied touch intensities
- Use **lower threshold** (0.6) for more notes
- Try different branches of plant

---

## 🎯 Testing Checklist

Try these to verify everything works:

- [ ] Server starts without errors
- [ ] Browser opens to http://localhost:5000
- [ ] Interface loads with purple gradient
- [ ] Can switch between Upload/Live modes
- [ ] Model selector has 3 options
- [ ] Threshold slider updates value display
- [ ] Can upload `touch_02.wav`
- [ ] Process button activates
- [ ] Notes play through speakers
- [ ] Piano keys light up
- [ ] Waveform displays
- [ ] Results list populates
- [ ] Can click piano keys manually

---

## 🚀 Next Steps

### **After Basic Testing**:

1. **Test All Your Recordings**
   - Try different touch intensities
   - Compare control vs touch files
   - Find optimal threshold

2. **Fine-Tune**
   - Adjust note ranges if needed
   - Try different models
   - Experiment with cooldown timing

3. **Show Others!**
   - Beautiful interface is demo-ready
   - Share with friends/colleagues
   - Record screen video

4. **Future Enhancements**:
   - Add recording capability
   - Implement live mode audio routing
   - Add more instruments
   - Save "melodies" (touch sequences)
   - Export detection data
   - Add statistics dashboard

---

## 💡 Cool Demo Ideas

### **Interactive Exhibition**
- Laptop + Plant SpikerBox + speakers
- Visitors touch plant, hear music
- Display showing interface on screen

### **Science Fair**
- Explain plant electrophysiology
- Show ML model working in real-time
- Compare different plants

### **Music Performance**
- "Play" the plant like an instrument
- Record touch sequences
- Create plant melodies

### **Educational Workshop**
- Teach biosignals + ML
- Students build their own models
- Compare results across different plants

---

## 📝 File Overview

**Backend** (`app.py`):
- Flask server
- SocketIO for real-time
- Model loading & prediction
- Audio processing
- REST API endpoints

**Frontend** (`templates/index.html`):
- Single-page application
- Responsive design
- Web Audio API for sound
- Canvas for visualization
- Socket.IO client

**Standalone** (previous scripts):
- `test_piano.py` - Terminal testing
- `train_model.py` - Model training
- `visualize_recordings.py` - Data inspection

---

## 🎉 You're Ready!

**Start the server**:
```bash
python app.py
```

**Open browser**:
```
http://localhost:5000
```

**Upload a file and make music!** 🌱🎹✨

---

## 📞 Common Questions

**Q: Can I use this with any plant?**
A: Yes! But you'll need to retrain the model with that plant's recordings.

**Q: Does it work on mobile?**
A: Yes! The interface is responsive. Open on phone/tablet browser.

**Q: Can I change the notes/scale?**
A: Yes! Edit the `NOTES` dictionary in `app.py` and `index.html`.

**Q: How accurate is it?**
A: Your model shows 100% on training data, expect 90-95% in real-world.

**Q: Can I add drums/other sounds?**
A: Absolutely! Modify the `playNote()` function in `index.html`.

---

**Enjoy your Plant Piano!** 🎵🌿

If something doesn't work, check the troubleshooting section or review the code comments.

Happy plant music making! 🎹✨
