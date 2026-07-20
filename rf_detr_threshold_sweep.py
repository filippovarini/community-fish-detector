#%% Header

"""
rf_detr_threshold_sweep.py

Run RF-DETR batch inference for multiple model/threshold combinations and
write a small summary table/report.

This is a smoke-test and threshold-comparison helper.  It does not train,
download weights, or produce accuracy/AP benchmark metrics.
"""

#%% Imports and constants

import argparse
import csv
import json
import math
import os
import re
import sys

from datetime import datetime

from rf_detr_batch_inference import run_detector_batch # type: ignore


SUMMARY_FIELDS = [
    'model',
    'checkpoint',
    'image_size',
    'threshold',
    'output_file',
    'entries',
    'failures',
    'entries_with_detections',
    'total_detections',
    'avg_detections_per_entry',
    'max_confidence'
]


#%% Argument parsing helpers

def _parse_name_value(value, value_name):
    """
    Parse a CLI value of the form name=value.
    """

    if '=' not in value:
        raise argparse.ArgumentTypeError(
            '{} must be in name=value format: {}'.format(value_name, value)
        )

    name, parsed_value = value.split('=', 1)
    name = name.strip()
    parsed_value = parsed_value.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            '{} is missing a name: {}'.format(value_name, value)
        )

    if not parsed_value:
        raise argparse.ArgumentTypeError(
            '{} is missing a value: {}'.format(value_name, value)
        )

    return name, parsed_value


def _parse_model_arg(value):
    """
    Parse --model name=checkpoint_path.
    """

    return _parse_name_value(value, 'model')


def _parse_image_size_arg(value):
    """
    Parse --image_size name=size.
    """

    name, image_size = _parse_name_value(value, 'image_size')

    try:
        image_size = int(image_size)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            'image_size must be an integer: {}'.format(value)
        )

    if image_size <= 0:
        raise argparse.ArgumentTypeError(
            'image_size must be positive: {}'.format(value)
        )

    return name, image_size


def _validate_unique_names(pairs, label):
    """
    Convert a list of (name, value) pairs to a dict, rejecting duplicates.
    """

    values_by_name = {}

    for name, value in pairs:
        if name in values_by_name:
            raise ValueError('Duplicate {} name: {}'.format(label, name))
        values_by_name[name] = value

    return values_by_name


def _sanitize_filename_component(value):
    """
    Convert a model name to a conservative filename component.
    """

    sanitized = re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip())
    sanitized = sanitized.strip('._-')
    if not sanitized:
        raise ValueError('Model name cannot be converted to a safe filename: {}'.format(value))
    return sanitized


def _format_threshold_for_filename(threshold):
    """
    Format thresholds consistently for deterministic output filenames.
    """

    return '{:.2f}'.format(threshold)


def _normalize_thresholds(thresholds):
    """
    Validate thresholds and reject duplicates.
    """

    normalized = []
    seen = set()

    for threshold in thresholds:
        if (not math.isfinite(threshold)) or (threshold < 0.0) or (threshold > 1.0):
            raise ValueError('Thresholds must be in [0, 1]: {}'.format(threshold))

        if threshold in seen:
            raise ValueError('Duplicate threshold: {}'.format(threshold))

        seen.add(threshold)
        normalized.append(threshold)

    return normalized


#%% Summary helpers

def summarize_md_output(output_path):
    """
    Summarize a MegaDetector-style JSON output file.
    """

    with open(output_path, 'r') as f:
        md_output = json.load(f)

    images = md_output.get('images', [])
    if images is None:
        images = []

    entries = len(images)
    failures = 0
    entries_with_detections = 0
    total_detections = 0
    max_confidence = None

    for entry in images:

        if entry.get('failure'):
            failures += 1

        detections = entry.get('detections', [])
        if detections is None:
            detections = []

        n_detections = len(detections)

        if n_detections > 0:
            entries_with_detections += 1
            total_detections += n_detections

        for detection in detections:
            if 'conf' not in detection:
                continue
            try:
                conf = float(detection['conf'])
            except (TypeError, ValueError):
                continue
            if (max_confidence is None) or (conf > max_confidence):
                max_confidence = conf

    avg_detections_per_entry = \
        float(total_detections) / entries if entries > 0 else 0.0

    return {
        'entries': entries,
        'failures': failures,
        'entries_with_detections': entries_with_detections,
        'total_detections': total_detections,
        'avg_detections_per_entry': avg_detections_per_entry,
        'max_confidence': max_confidence
    }


def _write_summary_csv(summary_rows, output_file):
    """
    Write threshold sweep summary rows to CSV.
    """

    with open(output_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _format_report_value(value):
    """
    Format values for Markdown table cells.
    """

    if value is None:
        return ''
    if isinstance(value, float):
        return '{:.4f}'.format(value).rstrip('0').rstrip('.')
    return str(value)


def _write_markdown_report(summary_rows, output_file):
    """
    Write a concise Markdown report for a threshold sweep.
    """

    with open(output_file, 'w') as f:
        f.write('# RF-DETR Threshold Sweep Report\n\n')
        f.write('Generated: {}\n\n'.format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
        f.write(
            'This report is a smoke-test and threshold-comparison helper. '
            'It is not a benchmark, and it does not produce accuracy, AP, or '
            'other ground-truth evaluation conclusions.\n\n'
        )

        f.write('| {} |\n'.format(' | '.join(SUMMARY_FIELDS)))
        f.write('| {} |\n'.format(' | '.join(['--'] * len(SUMMARY_FIELDS))))

        for row in summary_rows:
            values = [_format_report_value(row.get(field)) for field in SUMMARY_FIELDS]
            f.write('| {} |\n'.format(' | '.join(values)))


#%% Main sweep logic

def _validate_args(args, parser):
    """
    Validate parsed arguments and return normalized models/image sizes/thresholds.
    """

    if not os.path.exists(args.input):
        parser.error('Input path does not exist: {}'.format(args.input))

    if args.skip_images and args.skip_video:
        parser.error('--skip_images and --skip_video cannot both be true')

    if (args.frame_sample is not None) and (args.time_sample is not None):
        parser.error('--frame_sample and --time_sample cannot both be provided')

    if args.batch_size <= 0:
        parser.error('--batch_size must be positive')

    if args.loader_workers <= 0:
        parser.error('--loader_workers must be positive')

    if (args.frame_sample is not None) and (args.frame_sample <= 0):
        parser.error('--frame_sample must be positive')

    if (args.time_sample is not None) and (args.time_sample <= 0):
        parser.error('--time_sample must be positive')

    try:
        models = _validate_unique_names(args.model, 'model')
    except ValueError as e:
        parser.error(str(e))

    try:
        image_sizes = _validate_unique_names(args.image_size or [], 'image_size')
    except ValueError as e:
        parser.error(str(e))

    for model_name, checkpoint in models.items():
        if not os.path.isfile(checkpoint):
            parser.error('Checkpoint file does not exist for model {}: {}'.format(
                model_name, checkpoint
            ))

    for model_name in image_sizes:
        if model_name not in models:
            parser.error('--image_size provided for unknown model: {}'.format(model_name))

    try:
        thresholds = _normalize_thresholds(args.thresholds)
    except ValueError as e:
        parser.error(str(e))

    if len(thresholds) == 0:
        parser.error('At least one threshold is required')

    return models, image_sizes, thresholds


def run_threshold_sweep(args, parser):
    """
    Run all model/threshold combinations and write JSON, CSV, and Markdown outputs.
    """

    models, image_sizes, thresholds = _validate_args(args, parser)

    os.makedirs(args.output_dir, exist_ok=True)

    planned_output_files = []
    safe_model_names = {}

    for model_name in models:
        safe_model_name = _sanitize_filename_component(model_name)
        if safe_model_name in safe_model_names.values():
            parser.error('Model names collide after filename sanitization')
        safe_model_names[model_name] = safe_model_name

        for threshold in thresholds:
            output_basename = '{}_threshold_{}.json'.format(
                safe_model_name, _format_threshold_for_filename(threshold)
            )
            output_file = os.path.join(args.output_dir, output_basename)
            planned_output_files.append(output_file)

    summary_csv_file = os.path.join(args.output_dir, 'threshold_sweep_summary.csv')
    report_file = os.path.join(args.output_dir, 'threshold_sweep_report.md')
    planned_output_files.extend([summary_csv_file, report_file])

    if len(planned_output_files) != len(set(planned_output_files)):
        parser.error(
            'Output filename collision detected; use model names and thresholds that '
            'produce unique filenames'
        )

    if not args.overwrite:
        existing_files = [fn for fn in planned_output_files if os.path.exists(fn)]
        if len(existing_files) > 0:
            parser.error(
                'Output files already exist; pass --overwrite to replace them: {}'.format(
                    ', '.join(existing_files)
                )
            )

    summary_rows = []

    for model_name, checkpoint in models.items():

        image_size = image_sizes.get(model_name)

        for threshold in thresholds:

            output_file = os.path.join(
                args.output_dir,
                '{}_threshold_{}.json'.format(
                    safe_model_names[model_name],
                    _format_threshold_for_filename(threshold)
                )
            )

            print('\n*** Running model {} at threshold {} ***'.format(
                model_name, threshold
            ))

            _ = run_detector_batch(
                detector_file=checkpoint,
                image_folder=args.input,
                output_file=output_file,
                image_size=image_size,
                loader_workers=args.loader_workers,
                threshold=threshold,
                batch_size=args.batch_size,
                include_image_size=args.include_image_size,
                optimize_for_inference=args.optimize_for_inference,
                worker_type=args.worker_type,
                skip_images=args.skip_images,
                skip_video=args.skip_video,
                frame_sample=args.frame_sample,
                time_sample=args.time_sample,
                verbose=args.verbose
            )

            if not os.path.isfile(output_file):
                raise RuntimeError(
                    'Inference did not write an output file: {}'.format(output_file)
                )

            summary = summarize_md_output(output_file)
            row = {
                'model': model_name,
                'checkpoint': checkpoint,
                'image_size': image_size if image_size is not None else '',
                'threshold': _format_threshold_for_filename(threshold),
                'output_file': output_file,
                'entries': summary['entries'],
                'failures': summary['failures'],
                'entries_with_detections': summary['entries_with_detections'],
                'total_detections': summary['total_detections'],
                'avg_detections_per_entry': '{:.4f}'.format(
                    summary['avg_detections_per_entry']
                ),
                'max_confidence': '' if summary['max_confidence'] is None else \
                    '{:.4f}'.format(summary['max_confidence'])
            }
            summary_rows.append(row)

    _write_summary_csv(summary_rows, summary_csv_file)
    _write_markdown_report(summary_rows, report_file)

    print('\nSummary CSV written to {}'.format(summary_csv_file))
    print('Markdown report written to {}'.format(report_file))


def main():
    """
    Command-line driver.
    """

    parser = argparse.ArgumentParser(
        description='Run RF-DETR inference across model/threshold combinations and '
                    'summarize MegaDetector-format outputs.'
    )

    parser.add_argument(
        '--model',
        action='append',
        type=_parse_model_arg,
        required=True,
        help='Model checkpoint in name=checkpoint_path format; repeat for multiple models'
    )

    parser.add_argument(
        '--image_size',
        action='append',
        type=_parse_image_size_arg,
        default=[],
        help='Optional inference image size in name=size format; repeat as needed'
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to an image/video folder or a single image/video file'
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help='Directory where per-run JSON files and summary files will be written'
    )

    parser.add_argument(
        '--thresholds',
        nargs='+',
        type=float,
        required=True,
        help='One or more confidence thresholds to test'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output JSON, CSV, and report files'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for image inference (default: 1)'
    )

    parser.add_argument(
        '--loader_workers',
        type=int,
        default=4,
        help='Number of parallel image loader workers (default: 4)'
    )

    parser.add_argument(
        '--include_image_size',
        action='store_true',
        help='Include image dimensions in per-run JSON outputs'
    )

    parser.add_argument(
        '--skip_images',
        action='store_true',
        help='Ignore images and process only videos'
    )

    parser.add_argument(
        '--skip_video',
        action='store_true',
        help='Ignore videos and process only images'
    )

    parser.add_argument(
        '--frame_sample',
        type=int,
        default=None,
        help='Sample every Nth frame from videos'
    )

    parser.add_argument(
        '--time_sample',
        type=float,
        default=None,
        help='Sample frames every N seconds from videos'
    )

    parser.add_argument(
        '--worker_type',
        type=str,
        default='thread',
        choices=['thread', 'process'],
        help='Use threads or processes for image loading workers (default: thread)'
    )

    parser.add_argument(
        '--optimize_for_inference',
        action='store_true',
        help='Run optimize_for_inference() after loading each model'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable additional output from video processing'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    run_threshold_sweep(args, parser)


if __name__ == '__main__':
    main()
