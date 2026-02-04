#%% Header

"""
rf_detr_batch_inference.py

Run an RF-DETR detector on a folder of images, producing output in the
MegaDetector batch output format.

https://github.com/agentmorris/MegaDetector/tree/main/megadetector/api/batch_processing#megadetector-batch-output-format
"""

#%% Imports and constants

import argparse
import json
import os
import sys
import time
import torch

from datetime import datetime
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# RF-DETR model classes
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr import RFDETRNano

from megadetector.utils.ct_utils import round_float
from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS
from megadetector.utils.path_utils import find_images

# Mapping from model type strings to RF-DETR classes
MODEL_TYPE_MAP = {
    'nano': RFDETRNano,
    'base': RFDETRBase,
    'large': RFDETRLarge,
}

# By default, exclude detections below this confidence level
DEFAULT_CONFIDENCE_THRESHOLD = 0.005


#%% Support functions

def detect_model_type_from_checkpoint(checkpoint_path):
    """
    Detect the model type from a checkpoint file by inspecting its contents.

    Args:
        checkpoint_path (str): Path to .pth checkpoint file

    Returns:
        str: Model type string (e.g., 'nano', 'base', 'large')

    Raises:
        ValueError: If model type cannot be determined
    """

    print(f'Detecting model type from checkpoint: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if 'args' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'args' field. "
            f"Please specify --model_type explicitly."
        )

    args = checkpoint['args']

    if not hasattr(args, 'pretrain_weights'):
        raise ValueError(
            f"Checkpoint args does not contain 'pretrain_weights' field. "
            f"Please specify --model_type explicitly."
        )

    pretrain_weights = args.pretrain_weights
    print(f'Found pretrain_weights: {pretrain_weights}')

    # Extract model type from pretrain_weights string (e.g., "rf-detr-nano.pth" -> "nano")
    pretrain_weights_lower = pretrain_weights.lower()

    for model_type in MODEL_TYPE_MAP.keys():
        if model_type in pretrain_weights_lower:
            print(f'Detected model type: {model_type}')
            return model_type

    raise ValueError(
        f"Could not determine model type from pretrain_weights '{pretrain_weights}'. "
        f"Please specify --model_type explicitly. "
        f"Valid options: {list(MODEL_TYPE_MAP.keys())}"
    )

# ...def detect_model_type_from_checkpoint(...)


def load_image(image_path):
    """
    Load an image from disk.

    Args:
        image_path (str): Path to image file

    Returns:
        PIL.Image or None: Loaded image, or None if loading failed
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        return None


def load_images_batch(image_paths, num_workers=4):
    """
    Load a batch of images using multiple threads.

    Args:
        image_paths (list): List of image paths to load
        num_workers (int): Number of parallel workers

    Returns:
        list: List of (path, image) tuples. Image is None if loading failed.
    """

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        images = list(executor.map(load_image, image_paths))

    return list(zip(image_paths, images))


def convert_detections_to_md_format(detections, image_width, image_height):
    """
    Convert RF-DETR/Supervision detections to MegaDetector format.

    Args:
        detections: Supervision Detections object with xyxy, confidence, class_id
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels

    Returns:
        list: List of detection dicts in MegaDetector format
    """

    md_detections = []

    if detections is None or len(detections) == 0:
        return md_detections

    for i_detection in range(len(detections)):

        # Extract xyxy coordinates (absolute pixels)
        x1, y1, x2, y2 = detections.xyxy[i_detection]

        # Convert to normalized xywh format
        x_min_norm = float(x1) / image_width
        y_min_norm = float(y1) / image_height
        width_norm = float(x2 - x1) / image_width
        height_norm = float(y2 - y1) / image_height

        # Clamp values to [0, 1] range
        x_min_norm = max(0.0, min(1.0, x_min_norm))
        y_min_norm = max(0.0, min(1.0, y_min_norm))
        width_norm = max(0.0, min(1.0 - x_min_norm, width_norm))
        height_norm = max(0.0, min(1.0 - y_min_norm, height_norm))

        # Get confidence and class_id
        conf = float(detections.confidence[i_detection])

        # RF-DETR class_ids are 0-indexed when returned from the API
        class_id = int(detections.class_id[i_detection])

        # ...but we are loading class names from the model "class_names" dict,
        # which uses 1-indexed class IDs.  Increment to match.
        category = str(class_id + 1)

        bbox = round_float([x_min_norm, y_min_norm, width_norm, height_norm],
                           precision=COORD_DIGITS)
        conf = round_float(conf, precision=CONF_DIGITS)

        md_detections.append({
            'category': category,
            'conf': conf,
            'bbox': bbox
        })

    # ...for each detection

    return md_detections

# ...def convert_detections_to_md_format(...)


def run_detector_batch(
    detector_file,
    folder,
    output_file,
    image_size=640,
    loader_workers=4,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size=1,
    model_type=None,
    include_image_size=False,
    optimize_for_inference=False
):
    """
    Run RF-DETR detector on all images in a folder.

    Args:
        detector_file (str): Path to .pth checkpoint file
        folder (str): Path to folder containing images
        output_file (str): Path to output .json file
        image_size (int, optional): Image resolution for inference
        loader_workers (int, optional): Number of parallel image loaders
        threshold (float, optional): Confidence threshold for detections
        batch_size (int, optional): Batch size for inference
        model_type (str, optional): Model type ('nano', 'base', 'large') or None to auto-detect
        include_image_size (bool, optional): Whether to include image dimensions in output
        optimize_for_inference (bool, optional): Whether to optimize the model for inference,
            which should be a free lunch, but as of 9/2025 there is some risk of accuracy
            regression

    Returns:
        dict: Results dictionary in MegaDetector format
    """

    # Validate inputs
    assert os.path.isfile(detector_file), f'Detector file not found: {detector_file}'
    assert os.path.isdir(folder), f'Input folder not found: {folder}'
    assert output_file.endswith('.json'), 'Output file must have .json extension'

    # Determine model type
    if model_type is None:
        model_type = detect_model_type_from_checkpoint(detector_file)
    else:
        model_type = model_type.lower()
        if model_type not in MODEL_TYPE_MAP:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid options: {list(MODEL_TYPE_MAP.keys())}"
            )

    # Load model
    print(f'Loading {model_type} model from {detector_file}...')
    model_class = MODEL_TYPE_MAP[model_type]
    model = model_class(resolution=image_size, pretrain_weights=detector_file)
    print('Model loaded successfully')

    if optimize_for_inference:
        model.optimize_for_inference(batch_size=batch_size)
        # https://github.com/roboflow/rf-detr/issues/326#issuecomment-3321838797
        # model.optimize_for_inference(batch_size=batch_size,dtype=torch.bfloat16)

    # Get class names from model
    #
    # model.class_names is a dict mapping 1-indexed class IDs to names
    class_names = model.class_names
    print(f'Class names: {class_names}')

    # Build detection_categories dict (already 1-indexed string keys)
    detection_categories = {str(k): v for k, v in class_names.items()}

    # Find all images
    print(f'Searching for images in {folder}...')
    image_files = find_images(folder, recursive=True, return_relative_paths=False)
    print(f'Found {len(image_files)} images')

    if len(image_files) == 0:
        print('No images found, exiting')
        return None

    # Process images
    results = []
    start_time = time.time()

    # Process in batches
    for batch_start in tqdm(range(0, len(image_files), batch_size), desc='Processing'):

        batch_end = min(batch_start + batch_size, len(image_files))
        batch_paths = image_files[batch_start:batch_end]

        # Load images
        #
        # TODO: this is useful for testing batch inference, but it doesn't actually
        # parallelize inference and loading, i.e. the loading still blocks inference.
        batch_loaded = load_images_batch(batch_paths, num_workers=loader_workers)

        # Separate successful loads from failures
        valid_items = []
        for path, img in batch_loaded:
            rel_path = os.path.relpath(path, folder).replace('\\', '/')

            if img is None:
                # Failed to load image
                results.append({
                    'file': rel_path,
                    'failure': 'Image could not be loaded'
                })
            else:
                valid_items.append((path, rel_path, img))

        if len(valid_items) == 0:
            continue

        # Run inference
        images_for_inference = [item[2] for item in valid_items]

        try:
            if len(images_for_inference) == 1:
                detections_list = [model.predict(images_for_inference[0], threshold=threshold)]
            else:
                detections_list = model.predict(images_for_inference, threshold=threshold)
        except Exception as e:
            # If batch inference fails, mark all images in batch as failed
            print(f'Error during inference: {e}')
            for _, rel_path, _ in valid_items:
                results.append({
                    'file': rel_path,
                    'failure': f'Inference error: {str(e)}'
                })
            continue

        # Convert detections to MegaDetector format
        for (path, rel_path, img), detections in zip(valid_items, detections_list):
            img_width, img_height = img.size

            md_detections = convert_detections_to_md_format(
                detections, img_width, img_height
            )

            result = {
                'file': rel_path,
                'detections': md_detections
            }

            if include_image_size:
                result['width'] = img_width
                result['height'] = img_height

            results.append(result)

        # ...for each image

    # ...for each batch

    elapsed = time.time() - start_time
    images_per_second = len(image_files) / elapsed if elapsed > 0 else 0
    print(f'Processed {len(image_files)} images in {elapsed:.1f}s ({images_per_second:.2f} images/sec)')

    # Build output structure
    output = {
        'info': {
            'format_version': '1.5',
            'detector': os.path.basename(detector_file),
            'detection_completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detector_metadata': {
                'model_type': model_type,
                'image_size': image_size,
                'confidence_threshold': threshold
            }
        },
        'detection_categories': detection_categories,
        'images': results
    }

    # Write output
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'Results written to {output_file}')

    return output

# ...def run_detector_batch(...)


#%% Command-line interface

def main():

    parser = argparse.ArgumentParser(
        description='Run RF-DETR detector on a folder of images, producing MegaDetector-format output'
    )

    parser.add_argument(
        'detector_file',
        type=str,
        help='Path to RF-DETR checkpoint file (.pth)'
    )

    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing images (searched recursively)'
    )

    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output JSON file'
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=640,
        help='Image resolution for inference (default: 640)'
    )

    parser.add_argument(
        '--loader_workers',
        type=int,
        default=4,
        help='Number of parallel image loader workers (default: 4)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help='Confidence threshold for detections (default: {})'.format(
            DEFAULT_CONFIDENCE_THRESHOLD)
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['nano', 'base', 'large'],
        help='Model architecture type. If not specified, will attempt to auto-detect from checkpoint.'
    )

    parser.add_argument(
        '--include_image_size',
        action='store_true',
        help='Include image dimensions (width, height) in output'
    )

    parser.add_argument(
        '--optimize_for_inference',
        action='store_true',
        help='Run optimize_for_inference() after model load'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    run_detector_batch(
        detector_file=args.detector_file,
        folder=args.folder,
        output_file=args.output_file,
        image_size=args.image_size,
        loader_workers=args.loader_workers,
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_type=args.model_type,
        include_image_size=args.include_image_size,
        optimize_for_inference=args.optimize_for_inference
    )


if __name__ == '__main__':
    main()
