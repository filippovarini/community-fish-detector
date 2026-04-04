#%% Header

"""
convert_ckpt_to_pth.py

Convert PyTorch Lightning .ckpt checkpoints (as produced by the RF-DETR
training framework) to the .pth format expected by rf_detr_batch_inference.py
and by the rfdetr Python API (e.g. RFDETRLarge(pretrain_weights=...)).

Can be used as a CLI tool or imported as a module:

    from convert_ckpt_to_pth import convert_ckpt_to_pth
    convert_ckpt_to_pth(
        ckpt_path='checkpoint_49.ckpt',
        pth_path='checkpoint_49.pth',
        model_type='large',
        resolution=1024,
    )
"""

#%% Imports and constants

import argparse
import glob
import os
import sys
import tempfile

import torch


#%% Conversion function

def convert_ckpt_to_pth(
    ckpt_path,
    pth_path,
    model_type,
    resolution,
    class_names=None,
    strip_optimizer=True,
    overwrite=False
):
    """
    Convert a PyTorch Lightning .ckpt checkpoint to the .pth format used by
    the rfdetr inference API and rf_detr_batch_inference.py.

    Args:
        ckpt_path (str): Path to the input .ckpt file.
        pth_path (str): Path to the output .pth file.
        model_type (str): Model architecture type ('nano', 'small', 'base', 'medium', or 'large').
        resolution (int): Training image resolution (e.g. 560, 640, 728, 1024).
        class_names (list of str, optional): Class names for the model.  If None,
            will attempt to read from a checkpoint_best_total.pth or
            checkpoint_best_regular.pth in the same directory as the .ckpt file.
        strip_optimizer (bool): If True (default), exclude optimizer and scheduler
            state to produce a smaller file suitable for inference.
        overwrite (bool): If True, overwrite pth_path if it already exists.

    Returns:
        str: Path to the output .pth file.

    Raises:
        FileNotFoundError: If ckpt_path does not exist.
        FileExistsError: If pth_path already exists and overwrite is False.
        ValueError: If the .ckpt file does not contain a 'state_dict' key, or
            if model_type is not recognized.
    """

    valid_model_types = ('nano', 'small', 'base', 'medium', 'large')
    model_type = model_type.lower()
    if model_type not in valid_model_types:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {valid_model_types}"
        )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint file not found: {ckpt_path}')

    if os.path.exists(pth_path) and not overwrite:
        raise FileExistsError(
            f'Output file already exists: {pth_path}. '
            f'Use overwrite=True to replace it.'
        )

    print(f'Loading checkpoint: {ckpt_path}')
    checkpoint = torch.load(ckpt_path, weights_only=False, map_location='cpu')

    if 'state_dict' not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain 'state_dict' key. "
            f"Found keys: {sorted(checkpoint.keys())}. "
            f"This may not be a PyTorch Lightning checkpoint."
        )

    # Build the model dict by stripping the 'model.' prefix from state_dict keys
    state_dict = checkpoint['state_dict']
    model_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            model_dict[key[len('model.'):]] = value
        else:
            model_dict[key] = value

    # Try to find class_names from a companion .pth file if not provided
    if class_names is None:
        class_names = _find_class_names_from_companion(ckpt_path)
        if class_names is not None:
            print(f'Found class_names from companion .pth file: {class_names}')

    # Build the args dict with the fields that the inference script needs
    args = {
        'resolution': resolution,
        'model_type': model_type,
    }

    if class_names is not None:
        args['class_names'] = class_names

    epoch = checkpoint.get('epoch', None)
    if epoch is not None:
        args['epoch'] = epoch

    # Build the output checkpoint
    output = {
        'model': model_dict,
        'args': args,
    }

    if not strip_optimizer:
        if 'optimizer_states' in checkpoint:
            output['optimizer_states'] = checkpoint['optimizer_states']
        if 'lr_schedulers' in checkpoint:
            output['lr_schedulers'] = checkpoint['lr_schedulers']

    # Write atomically via temp file
    output_dir = os.path.dirname(os.path.abspath(pth_path))
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(output, tmp_path)
        os.replace(tmp_path, pth_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f'Wrote: {pth_path}')
    return pth_path


def _find_class_names_from_companion(ckpt_path):
    """
    Look for a companion .pth file in the same directory as the .ckpt file and
    extract class_names from its args.

    Args:
        ckpt_path (str): Path to the .ckpt file.

    Returns:
        list of str or None: Class names if found, else None.
    """

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))

    # Check for the standard "best" .pth files first, then any .pth file
    candidates = [
        os.path.join(ckpt_dir, 'checkpoint_best_total.pth'),
        os.path.join(ckpt_dir, 'checkpoint_best_regular.pth'),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                companion = torch.load(candidate, weights_only=False, map_location='cpu')
                args = companion.get('args', {})
                if isinstance(args, dict):
                    class_names = args.get('class_names', None)
                else:
                    class_names = getattr(args, 'class_names', None)
                if class_names is not None:
                    return class_names
            except Exception:
                continue

    return None


def convert_ckpt_folder_to_pth(
    ckpt_folder,
    model_type,
    resolution,
    class_names=None,
    strip_optimizer=True,
    overwrite=False,
    output_folder=None
):
    """
    Convert all .ckpt files in a folder to .pth format.

    Args:
        ckpt_folder (str): Path to folder containing .ckpt files.
        model_type (str): Model architecture type ('nano', 'small', 'base', 'medium', or 'large').
        resolution (int): Training image resolution.
        class_names (list of str, optional): Class names for the model.
        strip_optimizer (bool): If True (default), exclude optimizer state.
        overwrite (bool): If True, overwrite existing .pth files.
        output_folder (str, optional): Output folder for .pth files.  If None,
            writes to the same folder as the .ckpt files.

    Returns:
        list of str: Paths to the output .pth files.
    """

    if not os.path.isdir(ckpt_folder):
        raise FileNotFoundError(f'Folder not found: {ckpt_folder}')

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_folder, '*.ckpt')))
    if len(ckpt_files) == 0:
        print(f'No .ckpt files found in {ckpt_folder}')
        return []

    print(f'Found {len(ckpt_files)} .ckpt files in {ckpt_folder}')

    if output_folder is None:
        output_folder = ckpt_folder

    output_paths = []
    for ckpt_path in ckpt_files:
        basename = os.path.splitext(os.path.basename(ckpt_path))[0]
        pth_path = os.path.join(output_folder, basename + '.pth')
        try:
            convert_ckpt_to_pth(
                ckpt_path=ckpt_path,
                pth_path=pth_path,
                model_type=model_type,
                resolution=resolution,
                class_names=class_names,
                strip_optimizer=strip_optimizer,
                overwrite=overwrite,
            )
            output_paths.append(pth_path)
        except FileExistsError:
            print(f'Skipping (already exists): {pth_path}')
        except Exception as e:
            print(f'Error converting {ckpt_path}: {e}')

    print(f'Converted {len(output_paths)} / {len(ckpt_files)} checkpoints')
    return output_paths


#%% Command-line interface

def main():

    parser = argparse.ArgumentParser(
        description='Convert RF-DETR PyTorch Lightning .ckpt checkpoints to .pth format'
    )

    parser.add_argument(
        'input',
        type=str,
        help='Path to a .ckpt file or a folder containing .ckpt files'
    )

    parser.add_argument(
        'model_type',
        type=str,
        choices=['nano', 'small', 'base', 'medium', 'large'],
        help='Model architecture type'
    )

    parser.add_argument(
        'resolution',
        type=int,
        help='Training image resolution (e.g. 560, 640, 728, 1024)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output .pth file path (single file) or output folder (batch). '
             'Defaults to same location as input with .pth extension.'
    )

    parser.add_argument(
        '--class_names',
        type=str,
        nargs='+',
        default=None,
        help='Class names (e.g. --class_names fish bird). If not specified, '
             'will attempt to read from a companion .pth file.'
    )

    parser.add_argument(
        '--keep_optimizer',
        action='store_true',
        help='Keep optimizer and scheduler state in the output (default: strip them)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing .pth files'
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if os.path.isdir(args.input):
        convert_ckpt_folder_to_pth(
            ckpt_folder=args.input,
            model_type=args.model_type,
            resolution=args.resolution,
            class_names=args.class_names,
            strip_optimizer=not args.keep_optimizer,
            overwrite=args.overwrite,
            output_folder=args.output,
        )
    elif os.path.isfile(args.input):
        if args.output is None:
            args.output = os.path.splitext(args.input)[0] + '.pth'
        convert_ckpt_to_pth(
            ckpt_path=args.input,
            pth_path=args.output,
            model_type=args.model_type,
            resolution=args.resolution,
            class_names=args.class_names,
            strip_optimizer=not args.keep_optimizer,
            overwrite=args.overwrite,
        )
    else:
        print(f'Input not found: {args.input}')
        sys.exit(1)


if __name__ == '__main__':
    main()
