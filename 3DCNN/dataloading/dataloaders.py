import sys
import copy
from glob import glob
import math
import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
import nvvl

# Global label dict. Quite ugly!
label_dict = {}


def get_file_info(filename, frame_num, rand_changes):
    """
    Returns the base filename and frame_num for the inital fram in the loaded frame sequence

    Input:
        filename: name of the file where the sequence is loaded from
        frame_num: frame number of the first frame in the sequence
        rand_changes: changes made to the input image that may affect the label (e.g. for instance segmentation)

    Output:
        filename: name of the file where the sequence is loaded from
        frame_num: frame number of the first frame in the sequence
    """

    filename = os.path.basename(filename)

    return [filename, frame_num]

def get_label_file(filename, frame_num, rand_changes):
    """
    Loads labels from global label_dict, where there are a single label per video

    Input:
        filename: name of the file where the sequence is loaded from
        frame_num: frame number of the first frame in the sequence
        rand_changes: changes made to the input image that may affect the label (e.g. for instance segmentation)
    
    Output:
        List containing the labels, filename and frame number of the first frame in the sequence
    """

    dict_ind = label_dict[os.path.basename(filename)]

    return [dict_ind["labels"], filename, frame_num]


def get_label_minutes(filename, frame_num, rand_changes):
    """
    Loads labels from global label_dict, where there are labels per minute

    Input:
        filename: name of the file where the sequence is loaded from
        frame_num: frame number of the first frame in the sequence
        rand_changes: changes made to the input image that may affect the label (e.g. for instance segmentation)
    
    Output:
        List containing the labels, filename and frame number of the first frame in the sequence
    """

    dict_ind = label_dict[os.path.basename(filename)]
    offset = dict_ind["frameOffset"]    # How many frames left of the starting minute e.g. 16:00:45, has 15 seconds left 
                                        # This corresponds to 450 frames (30 FPS), and we assume we are halfway through the second, so 435 frame offset
                                        # These initial 435 frames are assigned to the label of 16:00:00, while the 436th label is assigned to 16:00:01
    FPM = dict_ind["FPM"]  # Frames per minute
    labels = dict_ind["labels"] # List of labels per minute

    # Logic flow to determine which label it should use (which minute)
    if frame_num < offset and offset > 0:
        # If there is an offset (i.e. the video starts in the middle of a minute) and the frame number is below this offset
        # return: Label of the first minute in the video
        ind = 0
    else:
        if offset > 0:
            # If there is an offset, and the frame is after the first minute
            # Subtract offset from frame number, and divide by FPM, and then round up to get our index. e.g.
            # offset = 300, frame_num = 400, FPM = 1800, ind = 400-100 / 1800 = 0.055 -> 1
            ind = int(np.ceil((frame_num-offset)/FPM))
        else:
            # If there is not an offset, and the frame is after the first minute
            # Take frame number, and divide by FPM, and then round down to get our index. e.g.
            # frame_num = 400, FPM = 1800,   ind = 400 / 1800 = 0.2222 -> 0
            ind = int(np.floor(frame_num/FPM))

    ind = min(ind, len(labels)-1)

    return [labels[ind], filename, frame_num]


def load_labels(filename):
    """
    Loads a JSON file containing the labels
    JSON data is saved into a global dict, and the function returns the corresponding function to get a specific label
    
    Input:
        filename: name of the JSON file
    
    Output:
        function based on the type of labels loaded
    """

    global label_dict
    print("Loading {}".format(filename))

    with open(filename) as data_file:
        label_dict = json.load(data_file)

    keys = list(label_dict.keys())
    tmp = label_dict[keys[0]]["labels"]

    if type(tmp) is list:
        return get_label_minutes
    elif type(tmp) is float:
        return get_label_file
    else:
        raise TypeError("The supplied JSON label file has stored labels as type {}, which is not supported".format(type(tmp)))


class RandomSubsetVidSampler(torch.utils.data.sampler.Sampler):
    """
    Samples a subset of elements randomly, without replacement.
    Subset size is determined by the share value, which should be betwen 0 and 1

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, share=0.5):
        self.data_source = data_source
        assert share > 0.0 and share <= 1.0, "Share input should be a value between 0 and 1"
        self.subset = int(len(data_source)*share)
        assert self.subset > 0, "The subset chosen is too small. Original Size: {}\nShare: {}\nSubset: {}".format(len(self.data_source), share, self.subset)

    def __iter__(self):
        return iter(np.random.choice(len(self.data_source),
                                     size=self.subset,
                                     replace=False).tolist())

    def __len__(self):
        return self.subset


class SequentialVidSampler(torch.utils.data.sampler.Sampler):
    """
    Sequentially samples the provided dataset

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, stride=8):
        self.data_source = data_source
        self.indecies = range(0, len(data_source), stride)
        
    def __iter__(self):
        return iter(self.indecies)

    def __len__(self):
        return len(self.indecies)


class RandomVidSampler(torch.utils.data.sampler.Sampler):
    """
    Randomly samples from the provided dataset

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, stride=8):
        self.data_source = data_source
        self.indecies = range(0, len(data_source), stride)

    def __iter__(self):
        return iter(self.indecies[i] for i in torch.randperm(len(self.indecies)))

    def __len__(self):
        return len(self.indecies)


class NVVL():
    def __init__(self, frames, is_cropped, crop_size, root,
                 batchsize = 1, device_id = 0,
                 shuffle = False, distributed = False, fp16 = False, 
                 index_map = None, random_flip = False, normalized = False, 
                 color_space = "RGB", dimension_order = "cfhw", 
                 get_label = None, sampler=None, stride=None):

        self.root = root
        self.batchsize = batchsize
        self.shuffle = shuffle
        self.distributed = distributed
        self.frames = frames
        self.device_id = device_id

        self.is_cropped = is_cropped
        self.crop_size = crop_size

        self.fp16 = fp16
        self.index_map = index_map

        self.random_flip = random_flip
        self.normalized = normalized
        self.color_space = color_space
        self.dimension_order = dimension_order

        self.get_label = get_label

        self.files = glob(os.path.join(self.root, '*.mp4'))

        if len(self.files) < 1:
            print(("[Error] No video files in %s" % (self.root)))
            raise LookupError

        if self.fp16:
            tensor_type = 'half'
        else:
            tensor_type = 'float'

        self.image_shape = nvvl.video_size_from_file(self.files[0])

        if self.is_cropped:
            self.height = self.crop_size[0]
            self.width = self.crop_size[1]
        else:
            self.height = self.image_shape.height
            self.width = self.image_shape.width

        print("Input size: {} x {}\nOutput size: {} x {}\n".format(self.image_shape.height, 
                                                                   self.image_shape.width,
                                                                   self.height,
                                                                   self.width))

        processing = {"input" : nvvl.ProcessDesc(type=tensor_type,
                                                 height=self.height,
                                                 width=self.width,
                                                 random_crop=self.is_cropped,
                                                 random_flip=self.random_flip,
                                                 normalized=self.normalized,
                                                 color_space=self.color_space,
                                                 dimension_order=self.dimension_order,
                                                 index_map=self.index_map),}

        dataset = nvvl.VideoDataset(self.files,
                                    sequence_length=self.frames,
                                    device_id=self.device_id,
                                    processing=processing,
                                    get_label=self.get_label)

        if sampler is not None and stride is not None:
            self.sampler = sampler(dataset, stride=stride)
        else:
            self.sampler = None

        self.loader = nvvl.VideoLoader(dataset,
                                       batch_size=self.batchsize,
                                       shuffle=self.shuffle,
                                       distributed=self.distributed,
                                       sampler=self.sampler)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)


def get_loader(args):
    """
    Sets up the dataloaders needed for training or testing

    Input:
        args: All input arguments in the main scrip
    
    Output:
        Dataloaders: The relvant dataloaders            
    """

    if args.test:
        fn = get_file_info

        args.shuffle = False

        print("Creating NVVL test dataloader")
        test_loader = NVVL(
            frames = args.frames,
            is_cropped = False,
            crop_size = args.crop_size,
            root = os.path.join(args.root, 'tst'),
            batchsize = args.val_batchsize,
            shuffle = False,
            distributed = False,
            device_id = 0,
            fp16 = False,
            random_flip = False,
            normalized = args.normalized,
            color_space = args.color_space,
            dimension_order = args.dimension_order,
            get_label = fn,
            sampler = SequentialVidSampler,
            stride = args.stride)

        test_batches = len(test_loader)
            
        sampler = None

        return test_loader, test_batches, sampler

    else:
        fn = load_labels(os.path.join(args.root, "labels", args.label_json))
        
        print("Creating NVVL training dataloader")
        train_loader = NVVL(
            frames = args.frames,
            is_cropped = args.is_cropped,
            crop_size = args.crop_size,
            root = os.path.join(args.root, 'train'),
            batchsize = args.batchsize,
            shuffle = args.shuffle,
            distributed = False,
            device_id = 0,
            fp16 = False,
            random_flip = args.random_flip,
            normalized = args.normalized,
            color_space = args.color_space,
            dimension_order = args.dimension_order,
            get_label = fn,
            sampler = RandomVidSampler,
            stride = args.stride)

        train_batches = len(train_loader)

        print("Creating NVVL validation dataloader")
        val_loader = NVVL(
            frames = args.frames,
            is_cropped = False,
            crop_size = args.crop_size,
            root = os.path.join(args.root, 'val'),
            batchsize = args.val_batchsize,
            shuffle = False,
            distributed = False,
            device_id = 0,
            fp16 = False,
            random_flip = False,
            normalized = args.normalized,
            color_space = args.color_space,
            dimension_order = args.dimension_order,
            get_label = fn,
            sampler = SequentialVidSampler,
            stride = args.stride)

        val_batches = len(val_loader)
            

        sampler = None

        return train_loader, train_batches, val_loader, val_batches, sampler