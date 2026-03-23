#%% Header

"""
rf_detr_batch_inference.py

Run an RF-DETR detector on a folder of images, producing output in the
MegaDetector batch output format.

http://lila.science/megadetector-output-format
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
import queue
import math

from multiprocessing import Process
from threading import Thread

# RF-DETR model classes
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr import RFDETRNano

from megadetector.utils.ct_utils import round_float, round_float_array
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
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

# Default maximum number of images to buffer ahead of inference
DEFAULT_MAX_QUEUE_SIZE = 20


#%% Support functions

def detect_model_info_from_checkpoint(checkpoint_path):
    """
    Detect model type and training resolution from a checkpoint file.

    Args:
        checkpoint_path (str): Path to .pth checkpoint file

    Returns:
        dict: Dictionary with keys:
            - 'model_type' (str): e.g. 'nano', 'base', 'large'
            - 'resolution' (int or None): training resolution, or None if not found

    Raises:
        ValueError: If model type cannot be determined
    """

    print(f'Reading checkpoint metadata from: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if 'args' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'args' field. "
            f"Please specify --model_type explicitly."
        )

    args = checkpoint['args']

    # Detect model type from pretrain_weights string
    if not hasattr(args, 'pretrain_weights'):
        raise ValueError(
            f"Checkpoint args does not contain 'pretrain_weights' field. "
            f"Please specify --model_type explicitly."
        )

    pretrain_weights = args.pretrain_weights
    print(f'Found pretrain_weights: {pretrain_weights}')

    pretrain_weights_lower = pretrain_weights.lower()

    detected_model_type = None
    for model_type in MODEL_TYPE_MAP.keys():
        if model_type in pretrain_weights_lower:
            detected_model_type = model_type
            break

    if detected_model_type is None:
        raise ValueError(
            f"Could not determine model type from pretrain_weights '{pretrain_weights}'. "
            f"Please specify --model_type explicitly. "
            f"Valid options: {list(MODEL_TYPE_MAP.keys())}"
        )

    print(f'Detected model type: {detected_model_type}')

    # Read training resolution if available
    resolution = getattr(args, 'resolution', None)
    if resolution is not None:
        print(f'Detected training resolution: {resolution}')

    return {
        'model_type': detected_model_type,
        'resolution': resolution
    }

# ...def detect_model_info_from_checkpoint(...)


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


def _producer_func(q, image_paths, image_folder):
    """
    Producer function for the image loading queue.

    Loads images from disk and puts (relative_path, PIL.Image) tuples onto a
    bounded queue.  Sends None when all images have been loaded.

    Args:
        q (queue.Queue): bounded queue shared with the consumer
        image_paths (list): absolute image file paths to load
        image_folder (str): root folder used to compute relative paths
    """

    for path in image_paths:
        rel_path = os.path.relpath(path, image_folder).replace('\\', '/')
        img = load_image(path)
        q.put((rel_path, img))

    # Signal that this producer is finished
    q.put(None)


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

    if (detections is None) or (len(detections) == 0):
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

        category = str(class_id)

        bbox = round_float_array([x_min_norm, y_min_norm, width_norm, height_norm],
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


#%% Batch inference function

def run_detector_batch(
    detector_file,
    image_folder,
    output_file,
    image_size=None,
    loader_workers=4,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size=1,
    model_type=None,
    include_image_size=False,
    optimize_for_inference=False,
    worker_type='thread'
):
    """
    Run RF-DETR detector on all images in a folder.

    Args:
        detector_file (str): Path to .pth checkpoint file
        image_folder (str): Path to folder containing images
        output_file (str): Path to output .json file
        image_size (int, optional): Image resolution for inference, None to load architecture
            default
        loader_workers (int, optional): Number of parallel image loaders
        threshold (float, optional): Confidence threshold for detections
        batch_size (int, optional): Batch size for inference
        model_type (str, optional): Model type ('nano', 'base', 'large') or None to auto-detect
        include_image_size (bool, optional): Whether to include image dimensions in output
        optimize_for_inference (bool, optional): Whether to optimize the model for inference,
            which should be a free lunch, but as of 9/2025 there is some risk of accuracy
            regression
        worker_type (str, optional): 'thread' or 'process' for image loading workers
            (default: 'thread')

    Returns:
        dict: Results dictionary in MegaDetector format
    """

    # Validate inputs
    assert os.path.isfile(detector_file), f'Detector file not found: {detector_file}'
    assert os.path.isdir(image_folder), f'Input folder not found: {image_folder}'
    assert output_file.endswith('.json'), 'Output file must have .json extension'

    # Determine model type and training resolution from checkpoint metadata
    checkpoint_info = detect_model_info_from_checkpoint(detector_file)

    if model_type is None:
        model_type = checkpoint_info['model_type']
    else:
        model_type = model_type.lower()
        if model_type not in MODEL_TYPE_MAP:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid options: {list(MODEL_TYPE_MAP.keys())}"
            )

    if image_size is None and checkpoint_info['resolution'] is not None:
        image_size = checkpoint_info['resolution']
        print(f'Using training resolution from checkpoint: {image_size}')

    # Load model
    print(f'Loading {model_type} model from {detector_file}...')
    model_class = MODEL_TYPE_MAP[model_type]
    if image_size is not None:
        model = model_class(resolution=image_size, pretrain_weights=detector_file)
        assert image_size == model.model_config.resolution, 'Model image size error'
    else:
        model = model_class(pretrain_weights=detector_file)
        image_size = model.model_config.resolution
        print('Using default architecture image size: {}'.format(image_size))

    print('Model loaded successfully')

    if optimize_for_inference:
        model.optimize_for_inference(batch_size=batch_size)
        # https://github.com/roboflow/rf-detr/issues/326#issuecomment-3321838797
        # model.optimize_for_inference(batch_size=batch_size,dtype=torch.bfloat16)

    # Get class names from model
    #
    # model.class_names is a list of strings.  Note to self: in older rfdetr versions, it was
    # a dict mapping 1-indexed class IDs to names.
    class_names = model.class_names
    print(f'Class names: {class_names}')

    # Build detection_categories dict
    detection_categories = {}
    for i_class,class_name in enumerate(class_names):
        detection_categories[str(i_class)] = class_name

    # Find all images
    print(f'Searching for images in {image_folder}...')
    image_files = find_images(image_folder, recursive=True, return_relative_paths=False)
    print(f'Found {len(image_files)} images')

    if len(image_files) == 0:
        print('No images found, exiting')
        return None

    # Process images using a producer/consumer pattern: loader threads populate a
    # bounded queue, and the main thread pulls from that queue for inference.
    results = []
    start_time = time.time()

    max_queue_size = max(DEFAULT_MAX_QUEUE_SIZE, 4 * batch_size)

    # Split image list across loader workers and start producer workers
    if worker_type == 'thread':
        image_queue = queue.Queue(maxsize=max_queue_size)
    else:
        import multiprocessing
        image_queue = multiprocessing.Queue(maxsize=max_queue_size)

    chunks = []
    chunk_size = math.ceil(len(image_files) / loader_workers)
    for i in range(loader_workers):
        chunk = image_files[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk) > 0:
            chunks.append(chunk)

    worker_class = Thread if worker_type == 'thread' else Process

    producers = []
    for chunk in chunks:
        t = worker_class(target=_producer_func, args=(image_queue, chunk, image_folder))
        t.daemon = True
        t.start()
        producers.append(t)

    # Consumer: pull loaded images from the queue and run inference in batches
    n_producers_finished = 0
    n_total_producers = len(producers)
    n_batches = math.ceil(len(image_files) / batch_size)

    pbar = tqdm(total=len(image_files), desc='Processing')

    while (n_producers_finished < n_total_producers):

        # Collect a batch of images from the queue
        valid_items = []  # list of (rel_path, img) for images that loaded successfully
        n_collected = 0

        while (n_collected < batch_size) and (n_producers_finished < n_total_producers):

            item = image_queue.get()

            # None is the sentinel indicating a producer thread has finished
            if item is None:
                n_producers_finished += 1
                continue

            rel_path, img = item
            n_collected += 1

            if img is None:
                results.append({
                    'file': rel_path,
                    'failure': 'Image could not be loaded'
                })
            else:
                valid_items.append((rel_path, img))

        # ...while collecting a batch

        pbar.update(n_collected)

        if len(valid_items) == 0:
            continue

        # Run inference
        images_for_inference = [item[1] for item in valid_items]

        try:
            if len(images_for_inference) == 1:
                detections_list = [model.predict(images_for_inference[0], threshold=threshold)]
            else:
                detections_list = model.predict(images_for_inference, threshold=threshold)
        except Exception as e:
            # If batch inference fails, mark all images in batch as failed
            print(f'Error during inference: {e}')
            for rel_path, _ in valid_items:
                results.append({
                    'file': rel_path,
                    'failure': f'Inference error: {str(e)}'
                })
            continue

        # Convert detections to MegaDetector format
        for (rel_path, img), detections in zip(valid_items, detections_list):
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

        # ...for each image in the batch

    # ...while producers are still running

    pbar.close()

    for t in producers:
        t.join()

    elapsed = time.time() - start_time
    images_per_second = len(image_files) / elapsed if elapsed > 0 else 0
    print(f'Processed {len(image_files)} images in {elapsed:.1f}s ({images_per_second:.2f} images/sec)')

    results = sort_list_of_dicts_by_key(results,'file')

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
        default=None,
        help='Image resolution for inference (default: None)'
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

    parser.add_argument(
        '--worker_type',
        type=str,
        default='thread',
        choices=['thread', 'process'],
        help='Use threads or processes for image loading workers (default: thread)'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    run_detector_batch(
        detector_file=args.detector_file,
        image_folder=args.folder,
        output_file=args.output_file,
        image_size=args.image_size,
        loader_workers=args.loader_workers,
        threshold=args.threshold,
        batch_size=args.batch_size,
        model_type=args.model_type,
        include_image_size=args.include_image_size,
        optimize_for_inference=args.optimize_for_inference,
        worker_type=args.worker_type
    )


if __name__ == '__main__':
    main()
