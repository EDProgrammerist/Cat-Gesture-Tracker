# Cat Gesture Tracker

A fun Python application that tracks your hand gestures and facial expressions using MediaPipe and displays corresponding cat meme reactions in real-time.

## Prerequisites
- Python 3.9+
- A webcam
- An internet connection the first time you run it (see note below)

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

1. Run the main script from your terminal:
   ```bash
   python main.py
   ```

   On the very first run, the script automatically downloads two small MediaPipe model files (`hand_landmarker.task` and `face_landmarker.task`) into the project folder. This requires an internet connection but only happens once — the files are cached locally and reused on subsequent runs.

2. Two windows will appear:
   - **Webcam Feed**: Shows your camera with hand landmarks.
   - **Cat Reaction**: Shows the currently matched cat meme.

3. **Gestures to try:**
   - **Shaka ("Call me")**: Extend your thumb and pinky, keep other fingers folded. -> Shows the rocking cat!
   - **Shh ("Finger on lips")**: Extend only your index finger, ideally near your mouth. -> Shows the silent cat!
   - **Mouth Wide Open**: Open your mouth wide (as a proxy for sticking your tongue out). -> Shows the licking cat!
   - **Default**: Do none of the above. -> Shows the standard polite cat.

   Gestures are checked in the order above, so a recognized hand gesture always takes priority over the mouth-open check. Only one hand and one face are tracked at a time.

4. To exit the application, click on the video window and press `q` or `ESC`.

## Troubleshooting

- **Webcam won't open**: The app tries camera indices 0-3 automatically. Make sure no other application (Zoom, Teams, the Camera app, etc.) is using the webcam, and that your OS privacy settings allow desktop apps to access the camera. On Windows, the app uses the DirectShow backend for more reliable camera access.
- **Model download fails**: If you're offline or behind a restrictive firewall on first run, the download of the `.task` model files will fail. Connect to the internet and re-run the script.
