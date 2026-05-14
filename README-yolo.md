## Running the YOLOv12x Community Fish Detector

### Clone the repo

```bash
git clone https://github.com/WildHackers/community-fish-detector.git
cd community-fish-detector
```

### Download the model weights

- You can find and download the `.pt` models from the [GitHub Releases]([url](https://github.com/WildHackers/community-fish-detector/releases))

### Install dependencies

```bash
pip install ultralytics
```

### Run inference

```python
from ultralytics import YOLO

# Load model
model = YOLO("path/to/your/model.pt")

# Run on an image or folder
results = model.predict(source="path/to/images_or_videos", imgsz=1024)

# Visualize results
results[0].show()
```

⚠️ Remember to set the image size to 1024 (`imgsz=1024`); YOLO inference tools default to an image size of 640.
