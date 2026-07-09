# Cat Gesture Tracker

A fun Python application that tracks your hand gestures and facial expressions using MediaPipe and displays corresponding cat meme reactions in real-time.

## Prerequisites
- Python 3.9+
- A webcam

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

2. Two windows will appear:
   - **Webcam Feed**: Shows your camera with hand landmarks.
   - **Cat Reaction**: Shows the currently matched cat meme.

3. **Gestures to try:**
   - **Shaka ("Call me")**: Extend your thumb and pinky, keep other fingers folded. -> Shows the rocking cat!
   - **Shh ("Finger on lips")**: Put your index finger near your mouth. -> Shows the silent cat!
   - **Mouth Wide Open**: Open your mouth wide (as a proxy for sticking your tongue out). -> Shows the licking cat!
   - **Default**: Do none of the above. -> Shows the standard polite cat.

4. To exit the application, click on the video window and press `q` or `ESC`.
