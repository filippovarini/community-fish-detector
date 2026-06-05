#%% Header

"""
rf_detr_batch_inference.py

Run an RF-DETR detector on a folder of images and/or videos (or a single image
or video file), producing output in the MegaDetector batch output format.

http://lila.science/megadetector-output-format

Videos are handled by sampling frames (by default one frame per second) and
running the detector on each sampled frame; per-frame detections are collected
into a single per-video entry in the output file, following the video
conventions described in the format spec.
"""

#%% Imports and constants

import argparse
import json
import os
import sys
import time
import torch

import numpy as np

from datetime import datetime
from PIL import Image
from tqdm import tqdm
import queue
import math

from multiprocessing import Process
from threading import Thread

# RF-DETR model classes
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium

from megadetector.utils.ct_utils import round_float, round_float_array
from megadetector.utils.ct_utils import sort_list_of_dicts_by_key
from megadetector.detection.run_detector import CONF_DIGITS, COORD_DIGITS
from megadetector.utils.path_utils import find_images

# Video support, all leveraged from the MegaDetector package
from megadetector.detection.video_utils import find_videos, is_video_file
from megadetector.detection.video_utils import run_callback_on_frames_for_folder
from megadetector.detection.video_utils import _filename_to_frame_number

# Mapping from model type strings to RF-DETR classes.
#
# 'base' is an older/internal name that predates the current public naming
# (nano, small, medium, large).  It's kept here for backward compatibility
# with legacy checkpoints whose pretrain_weights field contains 'base'.
MODEL_TYPE_MAP = {
    'nano': RFDETRNano,
    'small': RFDETRSmall,
    'base': RFDETRBase,
    'medium': RFDETRMedium,
    'large': RFDETRLarge,
}

# By default, exclude detections below this confidence level
DEFAULT_CONFIDENCE_THRESHOLD = 0.005

# Default maximum number of images to buffer ahead of inference
DEFAULT_MAX_QUEUE_SIZE = 20

# Default number of parallel image loading threads
DEFAULT_LOADER_WORKERS = 4

# When sampling frames from video and neither a frame interval nor a time
# interval is specified, sample frames at this rate (in seconds).  For a typical
# 30 fps video, this samples roughly every 30th frame.
DEFAULT_SECONDS_PER_VIDEO_FRAME = 1.0


#%% Support functions

def _ckpt_args_get(args, field, default=None):
    """
    Get a field from checkpoint args, handling both dict and Namespace formats.

    New checkpoints (PTL training stack) store args as a plain dict.
    Legacy checkpoints (pre-PTL engine) stored args as an argparse.Namespace.

    Args:
        args: The checkpoint['args'] value (dict or Namespace).
        field (str): Field name to retrieve.
        default: Value returned when the field is absent.

    Returns:
        The field value, or default if not found.
    """
    if isinstance(args, dict):
        return args.get(field, default)
    return getattr(args, field, default)


def detect_model_info_from_checkpoint(checkpoint_path):
    """
    Detect model type and training resolution from a checkpoint file.

    Supports both legacy .pth checkpoints (argparse.Namespace args with
    pretrain_weights) and new .pth checkpoints (dict args, as produced by the
    PTL training stack or convert_ckpt_to_pth.py).

    Args:
        checkpoint_path (str): Path to .pth checkpoint file

    Returns:
        dict: Dictionary with keys:
            - 'model_type' (str or None): e.g. 'nano', 'base', 'large', or None
              if not determinable
            - 'resolution' (int or None): training resolution, or None if not found

    Raises:
        ValueError: If a .ckpt file is passed instead of a .pth file
    """

    if checkpoint_path.lower().endswith('.ckpt'):
        raise ValueError(
            f"Cannot run inference directly from a .ckpt file: {checkpoint_path}\n"
            f"PyTorch Lightning .ckpt checkpoints must be converted to .pth format first.\n"
            f"Use convert_ckpt_to_pth.py to convert:\n"
            f"  python convert_ckpt_to_pth.py {checkpoint_path} <model_type> <resolution>"
        )

    print(f'Reading checkpoint metadata from: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')

    if 'args' not in checkpoint:
        print('Checkpoint does not contain args; model type and resolution must '
              'be specified explicitly.')
        return {'model_type': None, 'resolution': None}

    args = checkpoint['args']

    # Try to detect model type, checking both the 'model_type' field (new format,
    # written by convert_ckpt_to_pth.py) and the 'pretrain_weights' field (legacy format).
    detected_model_type = None

    # Check for explicit model_type field first (new format)
    model_type_field = _ckpt_args_get(args, 'model_type')
    if model_type_field is not None:
        model_type_lower = model_type_field.lower()
        if model_type_lower in MODEL_TYPE_MAP:
            detected_model_type = model_type_lower
            print(f'Found model_type in args: {detected_model_type}')

    # Fall back to pretrain_weights field (legacy format)
    if detected_model_type is None:
        pretrain_weights = _ckpt_args_get(args, 'pretrain_weights')
        if pretrain_weights is not None:
            print(f'Found pretrain_weights: {pretrain_weights}')
            pretrain_weights_lower = pretrain_weights.lower()
            for model_type in MODEL_TYPE_MAP.keys():
                if model_type in pretrain_weights_lower:
                    detected_model_type = model_type
                    break

    if detected_model_type is not None:
        print(f'Detected model type: {detected_model_type}')
    else:
        print('Could not determine model type from checkpoint args.')

    # Read training resolution if available
    resolution = _ckpt_args_get(args, 'resolution')
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


#%% Model loading

def load_model(detector_file,
               image_size=None,
               model_type=None,
               optimize_for_inference=False,
               batch_size=1):
    """
    Load an RF-DETR model from a checkpoint, auto-detecting the model type and
    training resolution when they aren't specified.

    Args:
        detector_file (str): Path to .pth checkpoint file
        image_size (int, optional): Image resolution for inference, None to use the
            resolution recorded in the checkpoint (or the architecture default)
        model_type (str, optional): Model type ('nano', 'small', 'base', 'medium',
            'large') or None to auto-detect from the checkpoint
        optimize_for_inference (bool, optional): Whether to optimize the model for
            inference, which should be a free lunch, but as of 9/2025 there is some
            risk of accuracy regression
        batch_size (int, optional): Batch size to pass to optimize_for_inference()

    Returns:
        dict: Dictionary with keys:
            - 'model': the loaded RF-DETR model
            - 'model_type' (str): resolved model type
            - 'image_size' (int): resolved inference resolution
            - 'detection_categories' (dict): mapping from string category IDs to class names
    """

    # Determine model type and training resolution from checkpoint metadata
    checkpoint_info = detect_model_info_from_checkpoint(detector_file)

    if model_type is None:
        model_type = checkpoint_info['model_type']
    else:
        model_type = model_type.lower()

    if model_type is None:
        raise ValueError(
            f"Could not determine model type from checkpoint. "
            f"Please specify --model_type explicitly. "
            f"Valid options: {list(MODEL_TYPE_MAP.keys())}"
        )
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

    return {
        'model': model,
        'model_type': model_type,
        'image_size': image_size,
        'detection_categories': detection_categories
    }

# ...def load_model(...)


#%% Image inference

def _run_detector_on_images(model,
                            image_files,
                            image_folder,
                            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                            batch_size=1,
                            loader_workers=DEFAULT_LOADER_WORKERS,
                            worker_type='thread',
                            include_image_size=False):
    """
    Run [model] on a list of image files, returning per-image results in
    MegaDetector format.

    Images are processed using a producer/consumer pattern: loader threads (or
    processes) populate a bounded queue, and the calling thread pulls from that
    queue for inference.

    Args:
        model: a loaded RF-DETR model (from load_model)
        image_files (list): absolute paths to the images to process
        image_folder (str): base folder used to compute relative output paths
        threshold (float, optional): confidence threshold for detections
        batch_size (int, optional): batch size for inference
        loader_workers (int, optional): number of parallel image loaders
        worker_type (str, optional): 'thread' or 'process' for image loading workers
        include_image_size (bool, optional): whether to include image dimensions in output

    Returns:
        list: per-image result dicts in MegaDetector format
    """

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

    pbar = tqdm(total=len(image_files), desc='Processing images')

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

    return results

# ...def _run_detector_on_images(...)


#%% Video inference

def _run_detector_on_videos(model,
                            video_folder,
                            video_files_relative,
                            threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                            frame_sample=None,
                            time_sample=None,
                            verbose=False):
    """
    Run [model] on a list of videos, returning one per-video result dict in
    MegaDetector format for each video.

    Frame extraction and sampling are handled by the MegaDetector package; we
    supply a per-frame callback that runs the detector on each sampled frame.
    Per-frame detections are collected into a single per-video entry, with a
    'frame_number' added to each detection and a sorted 'frames_processed' list,
    following the video conventions in the MegaDetector output format.

    Args:
        model: a loaded RF-DETR model (from load_model)
        video_folder (str): base folder used to compute relative output paths and to
            resolve [video_files_relative]
        video_files_relative (list): video paths relative to [video_folder]
        threshold (float, optional): confidence threshold for detections
        frame_sample (int, optional): process every Nth frame; mutually exclusive
            with time_sample
        time_sample (float, optional): process frames every N seconds; mutually
            exclusive with frame_sample
        verbose (bool, optional): enable additional debug output

    Returns:
        list: per-video result dicts in MegaDetector format
    """

    assert not ((frame_sample is not None) and (time_sample is not None)), \
        'frame_sample and time_sample are mutually exclusive'

    # The MegaDetector frame helpers use a single "every_n_frames" parameter, where
    # a negative value is interpreted as a sampling interval in seconds.
    if time_sample is not None:
        every_n_frames = -1 * time_sample
    else:
        every_n_frames = frame_sample

    start_time = time.time()

    def frame_callback(image_np, frame_id):
        """
        Run the detector on a single video frame.

        Args:
            image_np (numpy.ndarray): frame data in PIL orientation/channel order (RGB)
            frame_id (str): synthetic frame filename, e.g. "frame000030.jpg"

        Returns:
            dict: {'file': frame_id, 'detections': [...]} in MegaDetector format
        """

        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        frame_image = Image.fromarray(image_np)
        img_width, img_height = frame_image.size

        try:
            detections = model.predict(frame_image, threshold=threshold)
        except Exception as e:
            print(f'Error during inference on frame {frame_id}: {e}')
            return {'file': frame_id, 'detections': []}

        md_detections = convert_detections_to_md_format(
            detections, img_width, img_height)

        return {'file': frame_id, 'detections': md_detections}

    # ...def frame_callback(...)

    # [md_results] is a dict with keys 'video_filenames' (list of relative str),
    # 'frame_rates' (list of float), and 'results' (list, one element per video, of
    # lists of per-frame callback return values).  For failed videos, the frame rate
    # is -1 and 'results' is a dict with at least the key 'failure'.
    md_results = run_callback_on_frames_for_folder(
        input_video_folder=video_folder,
        frame_callback=frame_callback,
        every_n_frames=every_n_frames,
        verbose=verbose,
        files_to_process_relative=video_files_relative,
        error_on_empty_video=False)

    video_results = md_results['results']
    video_filenames = md_results['video_filenames']
    video_frame_rates = md_results['frame_rates']

    assert len(video_results) == len(video_filenames)
    assert len(video_results) == len(video_frame_rates)

    results = []

    # i_video = 0; results_this_video = video_results[i_video]
    for i_video, results_this_video in enumerate(video_results):

        video_fn = video_filenames[i_video]

        im = {}
        im['file'] = video_fn
        im['frame_rate'] = video_frame_rates[i_video]
        im['frames_processed'] = []

        if isinstance(results_this_video, dict):

            # This was a failed video
            assert 'failure' in results_this_video
            im['failure'] = results_this_video['failure']
            im['detections'] = None

        else:

            im['detections'] = []

            # results_one_frame = results_this_video[0]
            for results_one_frame in results_this_video:

                assert results_one_frame['file'].startswith(video_fn)

                frame_number = _filename_to_frame_number(results_one_frame['file'])

                assert frame_number not in im['frames_processed'], \
                    'Received the same frame twice for video {}'.format(im['file'])

                im['frames_processed'].append(frame_number)

                for det in results_one_frame['detections']:
                    det['frame_number'] = frame_number

                # This is a no-op if there were no above-threshold detections
                # in this frame
                im['detections'].extend(results_one_frame['detections'])

            # ...for each frame

        # ...was this a failed video?

        im['frames_processed'] = sorted(im['frames_processed'])

        results.append(im)

    # ...for each video

    elapsed = time.time() - start_time
    print(f'Processed {len(video_results)} videos in {elapsed:.1f}s')

    return results

# ...def _run_detector_on_videos(...)


#%% Batch inference function

def run_detector_batch(
    detector_file,
    image_folder,
    output_file,
    image_size=None,
    loader_workers=DEFAULT_LOADER_WORKERS,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size=1,
    model_type=None,
    include_image_size=False,
    optimize_for_inference=False,
    worker_type='thread',
    skip_images=False,
    skip_video=False,
    frame_sample=None,
    time_sample=None,
    verbose=False
):
    """
    Run RF-DETR detector on the images and/or videos in a folder, or on a single
    image or video file.

    Args:
        detector_file (str): Path to .pth checkpoint file
        image_folder (str): Path to a folder (searched recursively for images and/or
            videos) or to a single image or video file.  Despite the name, this may
            contain videos and/or be a single file; the name is retained for backward
            compatibility.
        output_file (str): Path to output .json file
        image_size (int, optional): Image resolution for inference, None to load architecture
            default
        loader_workers (int, optional): Number of parallel image loaders
        threshold (float, optional): Confidence threshold for detections
        batch_size (int, optional): Batch size for inference (images only; video frames
            are processed one at a time)
        model_type (str, optional): Model type ('nano', 'base', 'large') or None to auto-detect
        include_image_size (bool, optional): Whether to include image dimensions in output
            (images only)
        optimize_for_inference (bool, optional): Whether to optimize the model for inference,
            which should be a free lunch, but as of 9/2025 there is some risk of accuracy
            regression
        worker_type (str, optional): 'thread' or 'process' for image loading workers
            (default: 'thread')
        skip_images (bool, optional): ignore images, only process videos
        skip_video (bool, optional): ignore videos, only process images
        frame_sample (int, optional): sample every Nth frame from videos; mutually
            exclusive with time_sample
        time_sample (float, optional): sample frames every N seconds from videos;
            mutually exclusive with frame_sample.  If neither frame_sample nor
            time_sample is specified, defaults to DEFAULT_SECONDS_PER_VIDEO_FRAME.
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: Results dictionary in MegaDetector format
    """

    # Validate and normalize inputs
    assert os.path.isfile(detector_file), f'Detector file not found: {detector_file}'
    assert os.path.exists(image_folder), f'Input file/folder not found: {image_folder}'
    assert output_file.endswith('.json'), 'Output file must have .json extension'

    if loader_workers is None:
        loader_workers = DEFAULT_LOADER_WORKERS
    if threshold is None:
        threshold = DEFAULT_CONFIDENCE_THRESHOLD
    if batch_size is None:
        batch_size = 1
    if include_image_size is None:
        include_image_size = False
    if optimize_for_inference is None:
        optimize_for_inference = False
    if worker_type is None:
        worker_type = 'thread'

    if skip_images and skip_video:
        raise ValueError('Cannot skip both images and videos')

    if (frame_sample is not None) and (time_sample is not None):
        raise ValueError('frame_sample and time_sample are mutually exclusive')

    # Default the video sampling rate if the caller didn't specify one
    if (frame_sample is None) and (time_sample is None):
        time_sample = DEFAULT_SECONDS_PER_VIDEO_FRAME

    # Determine the set of images and videos to process, and the base folder used
    # to compute relative output paths.
    if os.path.isfile(image_folder):

        input_base_folder = os.path.dirname(image_folder)
        if is_video_file(image_folder):
            image_files = []
            video_files = [] if skip_video else [image_folder]
        else:
            image_files = [] if skip_images else [image_folder]
            video_files = []

    else:

        input_base_folder = image_folder

        if skip_images:
            image_files = []
        else:
            print(f'Searching for images in {image_folder}...')
            image_files = find_images(image_folder, recursive=True,
                                      return_relative_paths=False)
            print(f'Found {len(image_files)} images')

        if skip_video:
            video_files = []
        else:
            print(f'Searching for videos in {image_folder}...')
            video_files = find_videos(image_folder, recursive=True,
                                      return_relative_paths=False)
            print(f'Found {len(video_files)} videos')

    # ...whether the input is a file or a folder

    if (len(image_files) == 0) and (len(video_files) == 0):
        print('No images or videos found, exiting')
        return None

    # Load the model once; we'll use it for both images and videos
    model_info = load_model(detector_file,
                            image_size=image_size,
                            model_type=model_type,
                            optimize_for_inference=optimize_for_inference,
                            batch_size=batch_size)
    model = model_info['model']
    model_type = model_info['model_type']
    image_size = model_info['image_size']
    detection_categories = model_info['detection_categories']

    results = []

    # Process images
    if len(image_files) > 0:
        results.extend(_run_detector_on_images(
            model=model,
            image_files=image_files,
            image_folder=input_base_folder,
            threshold=threshold,
            batch_size=batch_size,
            loader_workers=loader_workers,
            worker_type=worker_type,
            include_image_size=include_image_size))

    # Process videos
    if len(video_files) > 0:
        video_files_relative = \
            [os.path.relpath(fn, input_base_folder).replace('\\', '/')
             for fn in video_files]
        results.extend(_run_detector_on_videos(
            model=model,
            video_folder=input_base_folder,
            video_files_relative=video_files_relative,
            threshold=threshold,
            frame_sample=frame_sample,
            time_sample=time_sample,
            verbose=verbose))

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
        description='Run RF-DETR detector on a folder of images and/or videos (or a '
                    'single image or video file), producing MegaDetector-format output'
    )

    parser.add_argument(
        'detector_file',
        type=str,
        help='Path to RF-DETR checkpoint file (.pth)'
    )

    parser.add_argument(
        'folder',
        type=str,
        help='Path to a folder containing images and/or videos (searched recursively), '
             'or to a single image or video file'
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
        choices=['nano', 'small', 'base', 'medium', 'large'],
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

    parser.add_argument(
        '--skip_images',
        action='store_true',
        help='Ignore images, only process videos'
    )

    parser.add_argument(
        '--skip_video',
        action='store_true',
        help='Ignore videos, only process images'
    )

    parser.add_argument(
        '--frame_sample',
        type=int,
        default=None,
        help='Sample every Nth frame from videos (mutually exclusive with --time_sample)'
    )

    parser.add_argument(
        '--time_sample',
        type=float,
        default=None,
        help='Sample frames every N seconds from videos (default: {}); mutually '
             'exclusive with --frame_sample'.format(DEFAULT_SECONDS_PER_VIDEO_FRAME)
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable additional debug output'
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
        worker_type=args.worker_type,
        skip_images=args.skip_images,
        skip_video=args.skip_video,
        frame_sample=args.frame_sample,
        time_sample=args.time_sample,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
