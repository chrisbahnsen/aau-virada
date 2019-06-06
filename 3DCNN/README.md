Rain estimation using 3D convolutional neural network
=====

This code base is for estimating whether rain is present in a sequence of images.
The code is based on PyTorch 0.4.0a0 and utilizes the [NVVL (NVIDIA Video Loader)](https://github.com/NVIDIA/nvvl) for loading sequenzes from video files. The code using the NVVL library was bsed on the [super-resolution example](https://github.com/NVIDIA/nvvl/tree/master/examples/pytorch_superres).


## Preparing data
In order to use NVVL data should be prepared in a correct manner. The data preparation steps taken are based on the recomendations from the NVVL team.

### Videos
The videos should be encoded as either H.264 or H.265, and support hardware decoding. This is done by using ffmpeg as follows.

H.264: ``` ffmpeg -i original.mp4 -map v:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -g X -profile:v high prepared.mp4```

H.265: ``` ffmpeg -i original.mp4 -map v:0 -c:v libx265 -crf 18 -pix_fmt yuv420p -x265-params "keyint=X:no-open-gop=1" -profile:v high prepared.mp4 ```

```-g X``` / ```-x265-params keyint=X:no-open-gop=1"``` indicates how many keyframes should be used when encoding the video. A rule.of.thumb is to set X equal to the lenght of sequences you are working with.

The videos should be saved using the .mp4 container. If they have been saved using e.g. the .mkv container, then NVVL might not be able to work with it. In order to change from e.g. .mkv to .mp4, just use this command:

``` ffmpeg -i original.mkv -c copy prepared.mp4```

If several files needs to have their container changed, this can be done with this single command line script:

```for i in *.mkv; do ffmpeg -i "$i" -c copy "/output/path/${i%.*}.mp4; done```

### Labels
In order to read labels for each sequence, it is necessary to supply a ```get_label(filename, frame_num, rand_changes)``` function to the NVVL VideoLoader object. 

This function gets called for each sample, with the filename and frame number of the first frame in the sequence, from which the correct label can be located.

We saved our labels as a JSON file, with each file being represented as a JSON object. If using a label per minute in the video, the objects would contain a list of labels, as well as a frameOffset and frame per minute (FPM) to locate the correct label for the provided frame. If a single label is used for an entire file, then the obejct just contains the single label.
Two ```get_label()``` functions were implemented to support this.

For the testing we simply retrun the filename and start frame, and saves the predictions of the network. The results are then analyzed by a set of separate scripts, found in the Analysis folder.


# Settings
The code base allows a wide array of settings, affecting how it is run, as well as model type/depth, hyperparameters etc.

## Models
The code currently supports the following models, shown as model [depth (if applicaple)] (implementation source).

* [C3D](https://arxiv.org/pdf/1412.0767.pdf) ([Davide Abati](https://github.com/DavideA/c3d-pytorch))


## NVVL Settings
NVVL supports the following augmentations which are done directly on the GPU:

* Scaling
* Random Crop
* Random Flip
* Color space (RGB or YCbCr)
* Normalization from [0; 255] to [0; 1]
* Index mapping (Determines order of the loaded frames in the sequence)

Furthermore, a custom PyTorch Sampler which only loads samples some user set percentage of all batches is available. To use this, set the parameter "subset" to a value between 0 and 1, and the sampler will select a random subset (equal to the percentage entered) of all batches.


## Analysing Tensorboard data

In order to analyse the tensorboard data two scripts are provided. *Read_tfEvents.py* reads the output summary file, stores it in a dict and saves it in a pickle file. This pickle file is then loaded by the *plotCNNData.py* script, where plots of the different tracked information is produced.