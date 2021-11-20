from datasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
from datasets.tube_crop import TubeCrop
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from utils.tube_utils import JSON_2_videoDetections
from utils.utils import get_number_from_string, AverageMeter, natural_sort
from tubes.run_tube_gen import extract_tubes_from_video

import torch
from torchvision import transforms
import numpy as np
import os

def train_regressor(
    _loader, 
    _epoch, 
    _model, 
    _criterion, 
    _optimizer, 
    _device, 
    _config, 
    _accuracy_fn=None, 
    _verbose=False):
    print('training at epoch: {}'.format(_epoch))
    _model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, data in enumerate(_loader):
        boxes, video_images, labels, num_tubes, paths, key_frames = data
        boxes, video_images = boxes.to(_device), video_images.to(_device)
        labels = labels.float().to(_device)
        key_frames = key_frames.to(_device)
       

        # video_images, labels, paths, key_frames, _ = data
        # video_images = video_images.to(_device)
        # labels = labels.to(_device)
        # key_frames = key_frames.to(_device)
        # boxes = None

        # print('video_images: ', video_images.size())
        # print('key_frames: ', key_frames.size())
        # print('boxes: ', boxes,  boxes.size())
        # print('labels: ', labels, labels.size())

        # zero the parameter gradients
        _optimizer.zero_grad()
        #predict
        outs = _model(video_images, key_frames, boxes, _config.num_tubes)
        #loss
        # print('labels: ', labels, labels.size(),  outs, outs.size())
        loss = _criterion(outs, labels)
        #accuracy
        if _accuracy_fn is not None:
            acc = _accuracy_fn(outs, labels)
        else:
            acc = 0
        
        # meter
        losses.update(loss.item(), outs.shape[0])
        accuracies.update(acc, outs.shape[0])
        # backward + optimize
        loss.backward()
        _optimizer.step()
        if _verbose:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    _epoch,
                    i + 1,
                    len(_loader),
                    loss=losses,
                    acc=accuracies
                )
            )
        
    train_loss = losses.avg
    train_acc = accuracies.avg
    print(
        'Epoch: [{}]\t'
        'Loss(train): {loss.avg:.4f}\t'
        'Acc(train): {acc.avg:.3f}'.format(_epoch, loss=losses, acc=accuracies)
    )
    return train_loss, train_acc

def prepare_tube_inputs(cfg, input_config, tubes_):
    sampler = TubeCrop(tube_len=cfg.NUM_FRAMES,
                        central_frame=True,
                        max_num_tubes=cfg.NUM_TUBES,
                        input_type=input_config['input_1'].itype,
                        sample_strategy=cfg.FRAMES_STRATEGY,
                        random=cfg.RANDOM,
                        box_as_tensor=False)
    sampled_frames_indices, chosed_tubes = sampler(tubes_)

def get_tube_scores(_model, _tubes, _device):
    """Get tube scores using a trained model. 

    Args:
        _model (cnn): trained model
        _tubes (list): list of action tubes
        _device (torch.device): hardware to eval

    Returns:
        list: List of tube scores
    """
    tube_scores=[]
    for i, tube in enumerate(_tubes):
        tube_real_numbers = [get_number_from_string(name) for name in tube['frames_name']]
        if tube['len']>3:
            #TODO
            images, bbox, keyframe = video_dataset.get_tube_data(
                tube, 
                get_number_from_string(frames_name[-1]), 
                get_number_from_string(frames_name[0]),
                0)
            images = images.to(_device)
            bbox = bbox.to(_device)
            keyframe = keyframe.to(_device)
            with torch.no_grad():
                outs = _model(images, keyframe, bbox, 1)
            tube_scores.append(outs.item())
        else:
            tube_scores.append(0)
    return tube_scores

def val_regressor(cfg, val_make_dataset, transformations, _model, _device, _epoch, _data_root):
    print('validation at epoch: {}'.format(_epoch))
    _model.eval()
    paths, labels, annotations, annotations_p_detections, num_frames = val_make_dataset()
    y_true = []
    pred_scores = []
    TUBE_BUILD_CONFIG['dataset_root'] = _data_root#'/media/david/datos/Violence DATA/UCFCrime2LocalClips/UCFCrime2LocalClips'

    for i, (path, label, annotation, annotation_p_detections, n_frames) in enumerate(zip(paths, labels, annotations, annotations_p_detections, num_frames)):
        video_dataset = UCFCrime2LocalVideoDataset(
            path=path,
            sp_annotation=annotation,
            transform=transforms.ToTensor(),
            clip_len=n_frames, #all frames
            clip_temporal_stride=1,
            transformations=transformations
        )
        frames_names = os.listdir(path)
        frames_names = natural_sort(frames_names)
        frames_indices = list(range(0,len(frames_names)))
        person_detections = JSON_2_videoDetections(annotation_p_detections) #load person detections
        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        for clip, frames_name, gt, num_frames in video_dataset:
            live_paths, time = extract_tubes_from_video(frames_indices, frames_names, MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG)
            
    