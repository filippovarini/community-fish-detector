#%% Header

"""

Simple test script for the Community Fish Detector.  Expects to be run from
the repo root.

"""

#%% Imports and constants

import os
import tempfile
import urllib.request

from megadetector.utils.path_utils import insert_before_extension
from megadetector.visualization.visualize_detector_output import visualize_detector_output
from megadetector.visualization.visualize_video_output import \
    visualize_video_output, VideoVisualizationOptions
from megadetector.utils.path_utils import open_file

from rf_detr_batch_inference import run_detector_batch # type: ignore

model_base = 'https://github.com/filippovarini/community-fish-detector/releases/download/2026.07.06-release/'
assert model_base.endswith('/')

model_urls = {}

model_urls['nano'] = model_base + 'cfd-rf-detr-nano-640-2026.02.02.cp-011.20260706-release.pth'
model_urls['small'] = model_base + 'cfd-rf-detr-small-1024-2026.06.06.cp-016.20260706-release.pth'
model_urls['medium'] = model_base + 'cfd-rf-detr-medium-1024-2026.03.24.cp-011.20260706-release.pth'

cfd_tmp_folder = os.path.join(tempfile.gettempdir(),'community-fish-detector-test')
test_image_folder = './test-images'

models_to_test = ['nano','small','medium']


#%% Test function

def test_community_fish_detector():

    preview_folders = []
    html_output_files = []

    for model_name in models_to_test:

        print('\n*** Testing model {} ***'.format(model_name))

        model_url = model_urls[model_name]

        # Create temporary folder if necessary
        os.makedirs(cfd_tmp_folder,exist_ok=True)

        assert os.path.isdir(test_image_folder), \
            'Test image folder not found, are you in the CFD repo root?'

        # If the URL is already a local file, no need to download it
        if os.path.isfile(model_url):
            model_file = model_url
            print('File {} exists, skipping download'.format(model_file))
        else:
            assert model_url.startswith('http'), 'Illegal model URL {}'.format(model_url)
            model_basename = os.path.basename(model_url)
            model_file = os.path.join(cfd_tmp_folder,model_basename)

            # Download model file
            if os.path.isfile(model_file):
                print('File {} already downloaded, skipping download'.format(model_file))
            else:
                print('Downloading model to {}'.format(model_file))
                urllib.request.urlretrieve(model_url,model_file)
                print('Download finished')

        output_file = os.path.join(cfd_tmp_folder,
                                   'cfd-test-output-{}.json'.format(model_name))

        # Add a timestamp to the filename
        output_file = insert_before_extension(output_file)

        print('Running model, writing results to {}'.format(output_file))

        # Run model, sampling videos at 4 fps (time_sample=0.25)
        _ = run_detector_batch(
                detector_file=model_file,
                image_folder=test_image_folder,
                output_file=output_file,
                image_size=None,
                time_sample=0.25
            )

        # Preview image results
        preview_folder = os.path.join(cfd_tmp_folder,'preview',model_name)
        html_output_file = os.path.join(preview_folder,'index.html')

        print('Finished running model, writing image results preview to {}'.format(preview_folder))

        visualize_detector_output(detector_output_path=output_file,
                                out_dir=preview_folder,
                                images_dir=test_image_folder,
                                confidence_threshold=0.25,
                                sample=-1,
                                output_image_width=1000,
                                random_seed=0,
                                html_output_file=html_output_file)

        print('Writing video results preview to {}'.format(preview_folder))

        video_options = VideoVisualizationOptions()
        video_options.confidence_threshold = 0.25

        visualize_video_output(detector_output_path=output_file,
                            out_dir=preview_folder,
                            video_dir=test_image_folder,
                            options=video_options)

        preview_folders.append(preview_folder)
        html_output_files.append(html_output_file)

    # ...for each model file

    for fn in html_output_files:
        open_file(fn)

# ...def def test_community_fish_detector(...)


#%% Command-line driver

if __name__ == '__main__':

    test_community_fish_detector()
