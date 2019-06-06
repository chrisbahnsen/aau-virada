import argparse
import logging as log
import os
import time
import shutil
import sys
import datetime


import numpy as np

from math import ceil, floor
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models as models

from dataloading.dataloaders import get_loader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()

## Parameters for NVVL loader (filepaths, augmentation settings)
parser.add_argument('--root', type=str, default='/root/3DCNN/',
                    help='input data root folder')
parser.add_argument('--output', type=str, default='',
                    help='output data root folder')
parser.add_argument('--label_json', type=str, default='labels_2hourlaserbinary.json',
                    help='JSON label filename')
parser.add_argument('--frames', type=int, default = 16,
                    help='num frames in input sequence (default: 16)')
parser.add_argument('--is_cropped', action='store_true',
                    help='crop input frames?')
parser.add_argument('--crop_size', type=int, nargs='+', default=[112, 112],
                    help='[height, width] for input crop (default: [112, 112])')
parser.add_argument('--shuffle', action="store_true",
                    help='Shuffle batches?')
parser.add_argument('--normalized', action="store_true",
                    help='Normalize images from [0;255] to [0;1]?')
parser.add_argument('--random_flip', action="store_true",
                    help='flip the image horizontally before cropping?')
parser.add_argument('--color_space', type = str, default = "RGB",
                    help='Color space to use. "RGB" and "YCbCr" are available. (default: "RGB")')
parser.add_argument('--dimension_order', type = str, default = "cfhw",
                    help='Axis order of the channels, frames, height and width. (default: "cfhw")')
parser.add_argument('--stride', type = int, default = None,
                    help='Frame stride when sampling from videos. (default: None)')
parser.add_argument('--test', action='store_true',
                    help='Whether to test a network, and not train it')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. If set, only 1 GPU is used')

## Hyperparameters
parser.add_argument('--batchsize', type=int, default=10,
                    help='Training batch size (default: 10)')
parser.add_argument('--val_batchsize', type=int, default=4,
                    help='validation/test batch size (default: 4)')
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--nesterov', action="store_true",
                    help='use Nesterov Accelerated Gradient')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--step_size', default=5, type=int,
                     help='Step size for lr schedueler (default: 5)')
parser.add_argument('--gamma', default=0.1, type=float,
                     help='Gamma for lr schedueler (default: 0.1)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run. (default: 90)')

## System settings
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


## Network parametes
parser.add_argument('--arch', metavar='ARCH', default='c3d',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: c3d)')
parser.add_argument("--num_classes", default=2, type=int, 
                    help="Number of neurons in output layer (if 1, sigmoid is last activation, otherwise softmax)")
parser.add_argument("--FCN", action="store_true",
                    help="Whether to use a dense validation/test approach. If not set center crop approach will be used")



def main(args):
    systemInfo()

    dirs = os.listdir(args.root)
    if args.test:
        assert "tst" in dirs, "A 'tst' directory is not in {}".format(args.root)
    else:
        assert "train" in dirs, "A 'train' directory is not in {}".format(args.root)
        assert "val" in dirs, "A 'val' directory is not in {}".format(args.root)
        assert "labels" in dirs, "A 'labels' directory is not in {}".format(args.root)

    del dirs

    if args.is_cropped:
        assert args.crop_size[0] == args.crop_size[1], "Crop size is assumed to be square, but you supplied {}".format(args.crop_size)
        args.sample_size = args.crop_size[0]

    args.sample_duration = args.frames

    if args.output == "":
        now = datetime.datetime.now()
        args.output = os.path.join("./results", now.strftime("%Y-%m-%d_%H:%M:%S"))
        del now

    if not os.path.exists(args.output):
        os.mkdir(args.output)
        os.mkdir(os.path.join(args.output, "weights"))

    print("Output path: {}".format(args.output))

    with open(os.path.join(args.output, "Settings.txt"), "w") as outfile:
        outfile.write(str(vars(args)))

    print("Setting up Tensorboard")
    writer = SummaryWriter()
    writer.add_text('config', str(vars(args)))
    print("Tensorboard set up")

    print("Setting Pytorch cuda settings")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    print("Set Pytorch cuda settings\n")

    print("Creating model '{}'".format(args.arch))
    model = load_model(args)
    print("Model created\n")

    if args.gpu is not None:
        print("Using GPU {}\n".format(args.gpu))
        model = model.cuda(args.gpu)
    elif torch.cuda.device_count() == 1:
        print("Using a single GPU\n")
        model = model.cuda()
    else:
        print("Using {} GPUs\n".format(torch.cuda.device_count()))
        model = nn.DataParallel(model).cuda()

    print("Setting up loss and optimizer")
    if args.num_classes == 1:
        criterion = nn.BCELoss().cuda(args.gpu)
    else:
        criterion = nn.NLLLoss().cuda(args.gpu)

    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=args.momentum,
                          nesterov = args.nesterov,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    print("Optimizer and loss function setup\n")

    best_accV = -1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print("Loading checkpoint from epoch {} with val accuracy of {}".format(checkpoint['epoch'], checkpoint['best_accV']))

            args.start_epoch = checkpoint['epoch']
            best_accV = checkpoint['best_accV']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})\n"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'\n".format(args.resume))



    if args.test:
        print("Initializing testing dataloaders") 
        test_loader, test_batches, sampler = get_loader(args)
        tst_samples_per_epoch = test_batches * args.test_batchsize

        print("Test Batch size: {}\nTest batches: {}\nTest videos: {}".format(args.test_batchsize, test_batches, len(test_loader.files)))
        print('Dataloaders initialized\n')

        # evaluate on validation set
        timeT = test(test_loader, model, args)

    else:
        print("Initializing training dataloaders") 
        train_loader, train_batches, val_loader, val_batches, sampler = get_loader(args)
        trn_samples_per_epoch = train_batches * args.batchsize
        val_samples_per_epoch = val_batches * args.val_batchsize
        print(args.root)
        print("Trn Batch size: {}\nTrn batches: {}\nTrn videos: {}\nVal Batch size: {}\nVal batches: {}\nVal videos: {}\nTrn samples per epoch: {}\nVals samples per epoch: {}".format(args.batchsize, train_batches, len(train_loader.files), args.val_batchsize, val_batches,len(val_loader.files), trn_samples_per_epoch, val_samples_per_epoch))
        print('Dataloaders initialized\n')

        for epoch in range(args.start_epoch, args.epochs):
                _start = time.time()
                
                scheduler.step()
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)

                # train for one epoch
                lossT, accT, timeT = train(train_loader, model, criterion, optimizer, epoch, writer, args)
                writer.add_scalar('Loss/Training-Avg', lossT, epoch)
                writer.add_scalar('Accuracy/Training', accT, epoch)
                writer.add_scalar('Time/Training-Avg', timeT, epoch)

                print("Epoch {} training completed: {}".format(epoch, datetime.datetime.now().isoformat()))
                print("Train time {}".format(timeT))
                time.sleep(1)

                # evaluate on validation set
                lossV, accV, timeV = validate(val_loader, model, criterion, args, epoch)
                writer.add_scalar('Loss/Validation-Avg', lossV, epoch)
                writer.add_scalar('Accuracy/Validation', accV, epoch)
                writer.add_scalar('Time/Validation-Avg', timeV, epoch)
                
                print("Epoch {} validation completed: {}".format(epoch, datetime.datetime.now().isoformat()))
                print("Val time {}".format(timeV))

                # remember best acc@1 and save checkpoint
                is_best = accV > best_accV
                best_accV = max(accV, best_accV)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_accV': best_accV,
                    'accV' :  accV,
                    'accT' : accT,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), dir=os.path.join(args.output, "weights"))

                _end = time.time()

                print("Epoch {}\n\tTime: {} seconds\n\tTrain Loss: {}\n\tTrain Accuracy: {}\n\tValidation Loss: {}\n\tValidation Accuracy: {}\n".format(epoch, _end-_start, lossT, accT, lossV, accV))

                print("Train time {}\nVal time {}".format(timeT, timeV))


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    """
    Takes the network and hyperparameters and trains the network through an iteration of the train data

    Input:
        train_loader: Dataloader for training data
        model: CNN model
        criterion: Loss function
        optimizer: Model optimizer function
        epoch: The current epoch
        writer: Tensorboard write
        args: General script arguments

    Output:
        losses.avg: Average loss value
        top1.avg: Average top-1 accuracy
        batch_time.avg: Average processign time per batch in seconds
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    n_batches = len(train_loader)

    end = time.time()
    for i, inputs in enumerate(train_loader):
        target = [x[0] for x in inputs['labels']]
        input = inputs['input']  # Output shape [batchsize, channels, numFrames, height, width]

        if args.num_classes == 1:
            target = torch.FloatTensor(target).view(-1, 1)
        else:
            target = torch.LongTensor(target).view(-1,)

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # compute output
        output = model(Variable(input))
        loss = criterion(output, Variable(target).cuda())

        # compute gradient and do optimizer step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        acc = accuracy(output, Variable(target).cuda())
        top1.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        writer.add_scalar('Loss/Training', loss.item(), epoch*n_batches+i)
        writer.add_scalar('Time/Training', batch_time.val, epoch*n_batches+i)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
    return losses.avg, top1.avg, batch_time.avg


def validate(val_loader, model, criterion, args, epoch):
    """
    Takes the network and hyperparameters and validates the network through an iteration of the validation data
    The predictions are saved in a csv file in a fodler 'val_predictions'

    Input:
        val_loader: Dataloader for validation data
        model: CNN model
        criterion: Loss function
        args: General script arguments
        epoch: The current epoch

    Output:
        losses.avg: Average loss value
        top1.avg: Average top-1 accuracy
        batch_time.avg: Average processign time per batch in seconds
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, inputs in enumerate(val_loader):
            target_full = inputs['labels']
            input = inputs['input']  # Output shape [batchsize, channels, numFrames, height, width]
            batchsize, _, _, height, width = input.shape
            input = input[:,:,:,:,:width-2]
            input = Variable(input)
            target = [x[0] for x in target_full]

            if args.num_classes == 1:
                target = torch.FloatTensor(target).view(-1, 1)
            else:
                target = torch.LongTensor(target).view(-1,)

            # Compute Output
            if args.FCN:
                #Fully Convolutional approach
                output = model(input)
            else:
                #Center crop approach
                c_w = width//2
                c_h = height//2
                h_w = args.crop_size[0]//2
                h_h = args.crop_size[1]//2
                output = model(input[:,:,:,c_h-h_h:c_h+h_h,c_w-h_w:c_w+h_w])
                                    
            loss = criterion(output, Variable(target).cuda())

            # measure accuracy and record loss
            losses.update(loss.item(), batchsize)

            acc = accuracy(output, Variable(target).cuda())
            top1.update(acc, batchsize)

            output = output.data
            pred = np.argmax(output,1)
            with open("./val_predictions/predictions_{}.csv".format(epoch), "a") as output_file:
                for j in range(len(target)):
                    output_file.write("{};{};{};{}\n".format(pred[j],output[j][pred[j]], target_full[j][1], target_full[j][2]))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))
    return losses.avg, top1.avg, batch_time.avg


def test(test_loader, model, args):
    """
    Takes the network and hyperparameters and tests the network on the test data
    The predictions are saved in a csv file in a fodler 'test_predictions'

    Input:
        test_loader: Dataloader for testing data
        model: CNN model
        args: General script arguments

    Output:
        batch_time.avg: Average processign time per batch in seconds
    """

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, inputs in enumerate(test_loader):
            target = inputs['labels']
            input = inputs['input']  # Output shape [batchsize, channels, numFrames, height, width]
            batchsize, _, _, height, width = input.shape
            input = input[:,:,:,:,:width-2]
            input = Variable(input)

            # Compute Output
            if args.FCN:
                #Fully Convolutional approach
                output = model(input)
            else:
                #Center crop approach
                c_w = width//2
                c_h = height//2
                h_w = args.crop_size[0]//2
                h_h = args.crop_size[1]//2
                output = model(input[:,:,:,c_h-h_h:c_h+h_h,c_w-h_w:c_w+h_w])

            output = output.data
            pred = np.argmax(output,1)
            with open("./test_predictions/predictions.csv", "a") as output_file:
                for j in range(len(target)):
                    output_file.write("{};{};{};{}\n".format(pred[j],output[j][pred[j]], target[j][0], target[j][1]))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                       i, len(test_loader), batch_time=batch_time))
    return batch_time.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', dir="./weights"):
    """
    Saves the current state of the network.

    Input:
        state: Dict of the model and other infromation which should be saved
        is_best: Boolean indicating if this is the best performance so far
        filename: Filename for the output pth.tar file
        dir: Path to the output directory
    """

    filename = os.path.join(dir, filename)
    torch.save(state, filename)
    if is_best: 
        shutil.copyfile(filename, os.path.join(dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the accuracy"""

    target = target.type(torch.cuda.LongTensor).view(-1,)

    with torch.no_grad():
        _, predicted = torch.max(output, 1)

        total = target.size(0)
        correct = predicted.eq(target).sum().item()

        res = correct / total
        
        return res


def load_model(args):

    if "c3d" in args.arch:
        model = models.__dict__[args.arch](num_classes=args.num_classes)
    else:
        raise ValueError("Supplied architecture {} is not supported".format(args.arch))

    print(str(model) + "\n")
    print("Total parameter count: {}".format(sum(p.numel() for p in model.parameters())))
    print("Trainable parameter count: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    return model


def systemInfo():
    from subprocess import call
    print('__Python VERSION: {}'.format(sys.version))
    print('__pyTorch VERSION: {}'.format(torch.__version__))
    print('__CUDA VERSION:')
    call(["nvcc", "--version"])

    print('__CUDNN VERSION: {}'.format(torch.backends.cudnn.version()))
    print('__Number CUDA Devices: {}'.format(torch.cuda.device_count()))
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])

    print('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    print('Available devices: {}'.format(torch.cuda.device_count()))
    print('Current cuda device: {}'.format(torch.cuda.current_device()))
    print()

if __name__ == "__main__":
    main(parser.parse_args())
