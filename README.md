# Community Fish Detector (CFD)

This repository provides pretrained object detection models for identifying one class: “fish”.

The model was trained on the [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset), a collaboratively built, large-scale dataset that unifies >1.9 million images and >935,000 fish bounding boxes from 17 open datasets spanning freshwater, marine, and lab environments.

With this project, our goal is to detect any fish, anywhere.

These models represent an initial training effort. They perform reasonably well across a variety of environments but can certainly be improved. If you’d like to contribute improvements or new experiments, [please get in touch](mailto:fppvrn@gmail.com)!


## Table of Contents

1. [Models](#models)
2. [Quick start](#quick-start)
3. [Contributors](#contributors)
4. [Citing this work](#citing-this-work)
5. [Example predictions](#example-predictions)
6. [Also see](#also-see)


## Models

| Model | Architecture | Input image size | Target classes | Training data | Inference license | AP |
|--|--|--|--|--|--|--|
| [cfd-2026.05.13-rf-detr-medium-1024](https://github.com/filippovarini/community-fish-detector/releases/download/2026.05.13-release/fish-detector-rf-detr-medium-1024-2026.03.24-checkpoint_11.stripped.pth) | [RF-DETR Medium](https://rfdetr.roboflow.com/reference/medium/) | 1024 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | Apache | .609 |
| [cfd-2026.05.13-rf-detr-small-1024](https://github.com/filippovarini/community-fish-detector/releases/download/2026.05.13-release/fish-detector-rf-detr-small-1024-2026.06.06-checkpoint_16.stripped.pth) | [RF-DETR Small](https://rfdetr.roboflow.com/reference/small/) | 1024 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | Apache | .606 |
| [cfd-2026.02.02-rf-detr-nano-640](https://github.com/filippovarini/community-fish-detector/releases/download/cfd-2026.02.02-rf-detr-nano/community-fish-detector-2026.02.02-rf-detr-nano-640.pth) | [RF-DETR Nano](https://rfdetr.roboflow.com/reference/nano/) | 640 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | Apache | .596 |
| [cfd-yolov12x-1.00](https://github.com/filippovarini/community-fish-detector/releases/download/cfd-1.00-yolov12x/cfd-yolov12x-1.00.pt) | [YOLOv12x](https://docs.ultralytics.com/models/yolo12/) | 1024 |  1 (fish) | [Community Fish Detection Dataset](https://lila.science/datasets/community-fish-detection-dataset) | AGPL | .588 |

The "inference license" column describes license information for the inference code you're likely to use when you run each model: the [RF-DETR Python package](https://pypi.org/project/rfdetr/) for RF-DETR models, and the [Ultralytics package](https://pypi.org/project/ultralytics/) for Ultralyics models.  This column does not describe the licenses associated with the [training data](https://lila.science/datasets/community-fish-detection-dataset), which is a composite of multiple datasets with a variety of licenses.

The "AP" column presents the average precision against the validation subset of the Community Fish Detection Dataset.  It's not straightforward to publish the splits we use (hopefully we will eventually, but we haven't done this yet), so you should not make anything of the absolute AP values.  This column is only useful to compare these models to each other and to future model releases.

## Quick start

These instructions describe the process for running RF-DETR versions of the CFD.

The YOLOv12x version is deprecated, and future models will be based on RF-DETR, because the AGPL license used by YOLOv12x and related model families is prohibitive for some of the use cases we'd like to support.  If you want to run the YOLOv12x version, see [README-yolo.md](README-yolo.md).


### Clone the repo

```bash
git clone https://github.com/filippovarini/community-fish-detector.git
cd community-fish-detector
```

### Download the model weights

You can download weights from the [Releases page]([url](https://github.com/filippovarini/community-fish-detector/releases)), or using the links in the summary table above.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run inference

#### Command-line batch inference

`rf_detr_batch_inference.py` runs the model recursively on a folder of images, and writes results in [MegaDetector output format](http://lila.science/megadetector-output-format).

```bash
python rf_detr_batch_inference.py "/path/to/your/model.pth" "/path/to/your/image/folder" "/path/to/your/output/file.json"
```

#### Programmatic inference

```python
from rfdetr import RFDETRNano
import supervision as sv
from PIL import Image

# Load model
model = RFDETRNano(pretrain_weights="/path/to/your/model.pth", resolution=640)

# Run on an image
image = Image.open("/path/to/your/image.jpg")
detections = model.predict(image, threshold=0.3)

# Annotate and save the result
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [f"fish {conf:.2f}" for conf in detections.confidence]

annotated = box_annotator.annotate(image.copy(), detections)
annotated = label_annotator.annotate(annotated, detections, labels=labels)
annotated.save("/path/to/your/output.jpg")
```

Be sure to specify the resolution (640 in this case, since this example uses the RF-DETR Nano model).  Specifying the inference resolution is not necessary when you're using the batch inference script, which automatically detects the training size.

#### Testing everything

You can test the whole pipeline as follows, assuming you are in a conda shell and you are in the repo root.  `test_community_fish_detector` will download weights for one of the models (defaults to the `medium` model), run it on the test images in this repo, and open a preview of the results in a browser.

```bash
# Create a conda environment
conda create -n cfd-inference python=3.12 pip -y && conda activate cfd-inference

# Install dependencies
pip install -r requirements.txt --upgrade --force --no-cache-dir

# Download weights, run the model, and generate a preview of the results
python test_community_fish_detector.py
```

## Contributors

This model was created by a collective effort of the following folks: <a href="https://www.linkedin.com/in/filippo-varini/">Filippo Varini</a>, <a href="https://dmorris.net">Dan Morris</a>, <a href="https://www.linkedin.com/in/sonny-burniston/">Sonny Burniston</a>, <a href="https://www.oceaneboulais.net/">Oceane Boulais</a>, <a href="https://www.mbari.org/person/kevin-barnard/">Kevin Barnard</a>, <a href="https://www.mbari.org/person/laura-chrobak/">Laura Chrobak</a>, <a href="https://alexvmt.github.io/">Alexander Merdian-Tarko</a>, <a href="https://www.linkedin.com/in/kameswari-devi-ayyagari-031820b7/">Devi Ayyagari</a>, <a href="https://www.linkedin.com/in/mona-dhiflaoui/">Mona Dhiflaoui</a>, <a href="https://www.linkedin.com/in/jiashu-chen-w/">Joshua Chen</a>, Gerard Calvo Bartra, Kalindi Fonda, Levi Veevee Cai, Giorgio De Pertis, Chris Jackett, Aditya Shirvalkar, Adrian Ibanez,  and many others.

If you contributed, but you don't see your name here, please [email us](mailto:fppvrn@gmail.com).

We welcome further contributions; if you have a dataset that could expand coverage, or want to contribute to improving the model, please [reach out](mailto:fppvrn@gmail.com)!

## Citing this work

If you use this model in your research or applications, please cite the GitHub repository:

```bibtex
@misc{community-fish-detector,
  author       = {Varini, Filippo and Morris, Dan and Burniston, Sonny and Boulais, Oceane and Barnard, Kevin and others},
  title        = {Community Fish Detector},
  year         = {2026},
  howpublished = {\url{https://github.com/filippovarini/community-fish-detector}},
}
```

Or in plain text:

> Varini, F., Morris, D., Burniston, S., Boulais, O., Barnard, K., et al. (2026). *Community Fish Detector*. GitHub. https://github.com/filippovarini/community-fish-detector

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
