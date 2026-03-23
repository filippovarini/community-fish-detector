# Community Fish Detector (CFD)

This repository provides pretrained object detection models for identifying one class: “fish”.

The model was trained on the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset), a collaboratively built, large-scale dataset that unifies >1.9 million images and >935,000 fish bounding boxes from 17 open datasets spanning freshwater, marine, and lab environments.

With this project, our goal is to detect any fish, anywhere. 

These models represent an initial training effort. They perform reasonably well across a variety of environments but can certainly be improved. If you’d like to contribute improvements or new experiments, [please get in touch](mailto:fppvrn@gmail.com)!


## Table of Contents

1. [Models](#models)  
2. [Quick start](#quick-start)  
3. [Contributors](#contributors)
4. [Example predictions](#example-predictions)  
5. [Also see](#also-see)  


## Models

| Model | Architecture | Input image size | Target classes | Dataset | License |
|--|--|--|--|--|--|
| [cfd-yolov12x-1.00.pt](https://github.com/WildHackers/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt) | [YOLOv12x](https://docs.ultralytics.com/models/yolo12/) | 1024 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | AGPL |

## Quick start

These instructions describe the process for running the RF-DETR version of the model.  The YOLOv12x version is deprecated, but you can see instructions for running it in an [older version of this README](https://github.com/filippovarini/community-fish-detector/tree/35564151a9f0f9c639ec5d0eb758fe35a64fa687?tab=readme-ov-file#quick-start).

### Clone the repo

```bash
git clone https://github.com/WildHackers/community-fish-detector.git
cd community-fish-detector
```

### Download the model weights

- You can find and download the `.pt` models from the [GitHub Releases]([url](https://github.com/WildHackers/community-fish-detector/releases))

### Install dependencies

```bash
pip install -r requirements.txt 
```

### Run inference

#### Command-line batch inference 

```bash
python rf_detr_batch_inference.py "/path/to/your/model.pt" "/path/to/your/image/folder" "/path/to/your/output/file.json" --image_size 640
```

Be sure to specify `--image_size 640`; the RF-DETR Nano architecture defaults to 384, but our detector was fine-tuned at 640, and we expect that you will get better results at 640.

#### Programmatic inference

```python
from rfdetr import RFDETRNano

# Load model
model = RFDetrNano(pretrain_weights="/path/to/your/model.pt", resolution=640)

# Run on an image or folder
results = model.predict(source="path/to/images_or_videos", imgsz=1024)

# Visualize results
results[0].show()
```

Be sure to specify `resolution=640`; the RF-DETR Nano architecture defaults to 384, but our detector was fine-tuned at 640, and we expect that you will get better results at 640.



## Contributors

This model was created by a collective effort of the following folks: <a href="https://www.linkedin.com/in/filippo-varini/">Filippo Varini</a>, <a href="https://dmorris.net">Dan Morris</a>, <a href="https://www.linkedin.com/in/sonny-burniston/">Sonny Burniston</a>, <a href="https://www.oceaneboulais.net/">Oceane Boulais</a>, <a href="https://www.mbari.org/person/kevin-barnard/">Kevin Barnard</a>, <a href="https://www.mbari.org/person/laura-chrobak/">Laura Chrobak</a>, <a href="https://alexvmt.github.io/">Alexander Merdian-Tarko</a>, <a href="https://www.linkedin.com/in/kameswari-devi-ayyagari-031820b7/">Devi Ayyagari</a>, <a href="https://www.linkedin.com/in/mona-dhiflaoui/">Mona Dhiflaoui</a>, <a href="https://www.linkedin.com/in/jiashu-chen-w/">Joshua Chen</a>, and many others.

If you contributed, but you don't see your name here, please [email us](mailto:fppvrn@gmail.com).

We welcome further contributions; if you have a dataset that could expand coverage, or want to contribute to improving the model, please [reach out](mailto:fppvrn@gmail.com)!

## Example Predictions

Below we provide some visual examples that overlay the ground truth with the model detections, to give you a qualitative sense of the model's training domain.

<img src="./assets/example7.png" />
<img src="./assets/example1.png" />
<img src="./assets/example2.png" />
<img src="./assets/example3.png" />
<img src="./assets/example4.png" />
<img src="./assets/example5.png" />
<img src="./assets/example6.png" />

## Also see

* [Hugging Face Space](https://huggingface.co/spaces/FathomNet/community-fish-detector) for this model set up by the [FathomNet](https://www.fathomnet.org/) community.
