# 2D Basketball Game with Hand Motion Control

A fun and interactive 2D basketball game that can be controlled using hand gestures through your camera, with fallback mouse and keyboard controls.

## Features

- **Hand Motion Control**: Use MediaPipe hand tracking to control the ball with your hand movements
- **Multiple Control Methods**: Hand tracking, mouse control, and keyboard shortcuts
- **Professional Graphics**: Enhanced basketball court with wood grain textures and realistic ball physics
- **Particle Effects**: Celebration effects when scoring baskets
- **Real-time Physics**: Realistic ball bouncing and trajectory
- **Score Tracking**: Keep track of your shots and accuracy percentage

## Requirements

- **Python 3.11 ONLY** (NOT 3.12, NOT 3.10 - EXACTLY 3.11!)
- Webcam (for hand tracking)
- Windows/Mac/Linux

## 🚀 Super Easy Setup (Anyone Can Do This!)

## ⚠️ IMPORTANT: You MUST Use Python 3.11 ONLY! ⚠️
**This game ONLY works with Python 3.11 - NOT 3.12, NOT 3.10, ONLY 3.11!**
- Download Python 3.11 from: https://www.python.org/downloads/release/python-3119/
- If you have a different Python version, this game will NOT work!

### Step 1: Download This Project 📥
- Click the **green "Code" button** on GitHub
- Click **"Download ZIP"**
- **Extract/Unzip** the folder to your Desktop

### Step 2: Open Command Prompt/Terminal 💻
**For Windows:**
- Press `Windows Key + R`
- Type `cmd` and press Enter
- A black window will open (this is good!)

**For Mac:**
- Press `Cmd + Space`
- Type `terminal` and press Enter

**For Linux:**
- Press `Ctrl + Alt + T`

### Step 3: Go to Your Project Folder 📁
Copy and paste this command (change the path if needed):
```bash
cd "C:\Users\YourName\Desktop\ping me"
```
*Replace "YourName" with your actual username*

### Step 4: Create a Safe Place for the Game 🏠
Copy and paste this EXACT command:
```bash
python -m venv basketball_env
```
*This creates a special folder for the game (like a toy box)*

### Step 5: Turn On the Toy Box 🔌
**For Windows** - Copy and paste this:
```bash
basketball_env\Scripts\activate
```

**For Mac/Linux** - Copy and paste this:
```bash
source basketball_env/bin/activate
```

*You'll see something like `(basketball_env)` appear - this means it worked!*

### Step 6: Install Game Parts 🎮
Copy and paste this command:
```bash
pip install -r requirements.txt
```
*This downloads all the game pieces (will take 1-2 minutes)*

### Step 7: PLAY THE GAME! 🏀
Copy and paste this final command:
```bash
python basketball_game.py
```

## 🎯 Quick Start (If You Get Lost)

**Just copy-paste these 4 commands one by one:**

```bash
cd "C:\Users\YourName\Desktop\ping me"
python -m venv basketball_env
basketball_env\Scripts\activate
pip install -r requirements.txt
python basketball_game.py
```

*Remember to change "YourName" to your actual Windows username!*

## 🔄 How to Play Again Later

**Every time you want to play:**
1. Open Command Prompt (`Windows Key + R`, type `cmd`)
2. Copy-paste these 3 commands:
```bash
cd "C:\Users\YourName\Desktop\ping me"
basketball_env\Scripts\activate
python basketball_game.py
```

## 😕 Having Problems?

**Problem: "python is not recognized"**
- Download Python 3.11 ONLY from [python.org](https://www.python.org/downloads/release/python-3119/)
- Check "Add Python to PATH" during installation
- DO NOT download Python 3.12 or any other version!

**Problem: Game crashes or won't start**
- Make sure you have Python 3.11 (check with `python --version`)
- If you have Python 3.12 or newer, uninstall it and install Python 3.11

**Problem: "No module named..."**
- Make sure you ran `pip install -r requirements.txt`
- Make sure you see `(basketball_env)` in your command prompt

**Problem: Camera not working**
- The game will still work with mouse control!
- Try running as administrator

## Controls

### Hand Control Mode (Primary)
- Move your hand in front of the camera to position the ball
- The game automatically detects your hand and follows its movement

### Mouse Control Mode (Fallback)
- **Click**: Position the ball at mouse cursor location
- Automatically switches to mouse mode if camera is not available

### Keyboard Shortcuts
- **Spacebar**: Shoot the ball towards the hoop
- **R**: Reset ball to starting position
- **Esc**: Quit the game

## Game Features

- **Enhanced Visuals**: Professional basketball court with wood grain textures
- **Ground Bar**: Realistic floor surface for ball bouncing
- **Smart UI**: Compact control panel positioned to not interfere with gameplay
- **Particle System**: Visual effects for successful shots
- **Statistics**: Real-time tracking of score, shots taken, and accuracy percentage

## Technical Details

- **Computer Vision**: MediaPipe for hand landmark detection
- **Graphics**: Pygame for 2D rendering and game loop
- **Physics**: Custom ball physics with realistic bouncing and gravity
- **Performance**: Optimized for smooth 60 FPS gameplay

## Troubleshooting

- **Camera not working**: The game will automatically switch to mouse control mode
- **Hand tracking not responsive**: Ensure good lighting and clear hand visibility
- **Performance issues**: Close other applications that might be using the camera

## Project Structure

```
ping me/
├── basketball_game.py     # Main game file
├── requirements.txt       # Project dependencies
├── basketball_env/        # Virtual environment (auto-created)
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

## Dependencies

- **pygame**: Game engine and graphics
- **opencv-python**: Computer vision and camera access
- **mediapipe**: Hand tracking and gesture recognition
- **numpy**: Mathematical operations and array handling

## Contributing

Feel free to fork this project and add your own features! Some ideas:
- Different difficulty levels
- Multiple basketballs
- Power-ups and special effects
- Multiplayer support
- Sound effects and music

## License

This project is open source and available under the MIT License.

---

Enjoy playing basketball with hand gestures! 🏀
