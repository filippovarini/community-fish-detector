#%% Header

"""
prepare_pth_for_inference.py

Strip a .pth checkpoint and ensure it contains the metadata needed by
rf_detr_batch_inference.py (model_type and resolution).

Can be used as a CLI tool or imported as a module:

    from prepare_pth_for_inference import prepare_pth_for_inference
    prepare_pth_for_inference(
        input_path='checkpoint_best_total.pth',
        output_path='checkpoint_best_total.stripped.pth',
        model_type='large',
        resolution=1024,
    )
"""

#%% Imports and constants

import argparse
import os
import shutil
import sys

import torch

from rfdetr.utilities.state_dict import strip_checkpoint

VALID_MODEL_TYPES = ('nano', 'small', 'base', 'medium', 'large')


#%% Main function

def prepare_pth_for_inference(
    input_path,
    output_path,
    model_type,
    resolution,
    class_names=None,
):
    """
    Strip a .pth checkpoint and add model_type and resolution to args if
    not already present.

    Uses rfdetr.utilities.state_dict.strip_checkpoint to remove optimizer
    and scheduler state, then adds metadata fields needed by the inference
    script.

    Args:
        input_path (str): Path to the input .pth file.
        output_path (str): Path to the output .pth file.  Must be different
            from input_path.
        model_type (str): Model architecture type ('nano', 'small', 'base',
            'medium', or 'large').
        resolution (int): Training image resolution (e.g. 560, 640, 728, 1024).
        class_names (list of str, optional): Class names for the model.  Only
            added to args if not already present.

    Returns:
        str: Path to the output .pth file.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ValueError: If input_path and output_path resolve to the same file,
            or if model_type is not recognized.
    """

    model_type = model_type.lower()
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {VALID_MODEL_TYPES}"
        )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f'Input file not found: {input_path}')

    input_resolved = os.path.realpath(input_path)
    output_resolved = os.path.realpath(output_path)
    if input_resolved == output_resolved:
        raise ValueError(
            f'Input and output paths resolve to the same file: {input_resolved}'
        )

    # Copy input to output, then strip in place (matching strip_checkpoint's
    # in-place API)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy2(input_path, output_path)

    print(f'Stripping checkpoint: {output_path}')
    strip_checkpoint(output_path)

    # Add metadata fields to args
    checkpoint = torch.load(output_path, weights_only=False, map_location='cpu')
    args = checkpoint.get('args', {})

    if not isinstance(args, dict):
        # Legacy Namespace format; convert to dict
        args = vars(args)

    modified = False

    if 'model_type' not in args:
        args['model_type'] = model_type
        modified = True
        print(f'Added model_type: {model_type}')
    else:
        existing = args['model_type'].lower()
        if existing != model_type:
            raise ValueError(
                f"model_type mismatch: checkpoint has '{existing}' "
                f"but '{model_type}' was specified"
            )
        print(f'model_type already present: {existing}')

    if 'resolution' not in args:
        args['resolution'] = resolution
        modified = True
        print(f'Added resolution: {resolution}')
    else:
        existing = args['resolution']
        if existing != resolution:
            raise ValueError(
                f"resolution mismatch: checkpoint has {existing} "
                f"but {resolution} was specified"
            )
        print(f'resolution already present: {existing}')

    if class_names is not None and 'class_names' not in args:
        args['class_names'] = class_names
        modified = True
        print(f'Added class_names: {class_names}')

    if modified:
        checkpoint['args'] = args
        torch.save(checkpoint, output_path)

    print(f'Wrote: {output_path}')
    return output_path


#%% Command-line interface

def main():

    parser = argparse.ArgumentParser(
        description='Strip an RF-DETR .pth checkpoint and add inference metadata'
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to the input .pth file'
    )

    parser.add_argument(
        'output',
        type=str,
        help='Path to the output .pth file (must differ from input)'
    )

    parser.add_argument(
        'model_type',
        type=str,
        choices=list(VALID_MODEL_TYPES),
        help='Model architecture type'
    )

    parser.add_argument(
        'resolution',
        type=int,
        help='Training image resolution (e.g. 560, 640, 728, 1024)'
    )

    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=None,
        help='Class names (e.g. --class_names fish bird). Only added if '
             'not already present in the checkpoint.'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    prepare_pth_for_inference(
        input_path=args.input,
        output_path=args.output,
        model_type=args.model_type,
        resolution=args.resolution,
        class_names=args.class_names,
    )


if __name__ == '__main__':
    main()
