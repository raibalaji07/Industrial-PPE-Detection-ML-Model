# Industrial PPE Detection ML Model

This project utilizes machine learning models to detect and ensure the presence of personal protective equipment (PPE) in an industrial setting, focusing on items such as helmets, jackets, goggles, gloves, and footwear. The system integrates computer vision techniques with deep learning models for real-time detection and alerting.

## Features

- **Object Detection**: Utilizes Ultralytics YOLO models to detect persons and various safety equipment items.
- **Real-time Monitoring**: Monitors a video feed (e.g., from a CCTV camera) to identify individuals and verify their PPE usage.
- **Alert Mechanism**: Alerts users when necessary PPE items (like helmets and jackets) are missing, providing visual cues in the video feed.
- **Score Reporting**: Provides confidence scores for detected safety equipment items.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/raibalaji07/Industrial-PPE-Detection-ML-Model.git
   cd Industrial-PPE-Detection-ML-Model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the required YOLO model weights:
   - `yolov8l.pt` for person detection
   - `best.pt` for safety equipment detection

2. Update the paths in the Python script to point to these model weights and your video source.

3. Run the Python script:
   ```bash
   ppefinal.py
   ```

4. Press `q` to exit the video feed.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pygame (`playsound` for optional sound alerts)
- Ultralytics YOLO

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---