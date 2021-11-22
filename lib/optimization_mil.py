from datasets.ucfcrime2local_dataset import UCFCrime2LocalVideoDataset
from datasets.tube_crop import TubeCrop
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from utils.tube_utils import JSON_2_videoDetections
from utils.utils import get_number_from_string, AverageMeter, natural_sort
from tubes.run_tube_gen import extract_tubes_from_video
from lib.spatio_temp_iou import st_iou

import torch
import numpy as np

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

def get_tube_scores(_model, _video_images, _key_frames, _boxes, _device):
    """Get tube scores using a trained model. 

    Args:
        _model (cnn): trained model
        _video_images (torch.tensor): Tensor of shape [N,C,T,W,H] where N is the number of tubes, C chanels, T temporal dim, W width, H heigh.
        _key_frames (torch.tensor): Tensor of shape [N,C,,W,H] where N is the number of keyframes, C chanels, W width, H heigh.
        _boxes (torch.tensor): Tensor of shape [N,5] where N is the number of boxes, and 5 the id and four box coordinates.
        _device (torch.device): hardware to eval

    Returns:
        list: List of tube scores
    """
    tube_scores=[]
    for i in range(_video_images.size(0)): #iterate tube per tube
        tube_images = torch.unsqueeze(_video_images[i], dim=0).to(_device)
        tube_bbox = torch.unsqueeze(_boxes[i], dim=0).to(_device)
        tube_keyframe = torch.unsqueeze(_key_frames[i], dim=0).to(_device)
        with torch.no_grad():
            try:
                outs = _model(tube_images, tube_keyframe, tube_bbox, 1)
            except Exception as e:
                print("\nOops!", e.__class__, "occurred.")
                print("tube_images: ", tube_images.size())
                print("tube_key_frame: ", tube_keyframe.size())
                print("tube_bbox: ", tube_bbox, tube_bbox.size())
                exit()
        tube_scores.append(outs.item())
    return tube_scores

def max_tube_idx(tube_scores):
    tube_scores = np.array(tube_scores)
    max_idx = np.argmax(tube_scores)
    return max_idx

def val_regressor(cfg, val_make_dataset, transformations, _model, _device, _epoch, _data_root):
    """[summary]

    Args:
        cfg (yaml): cfg.TUBE_DATASET
        val_make_dataset ([type]): [description]
        transformations (dict{input_1: CnnInputConfig, input_2: CnnInputConfig}): Dictionary of input configurations
        _model (cnn): Model
        _device (torch.device): Pytorch device
        _epoch (int): Epoch
        _data_root (str): Path to datasets folder
    """
    print('validation at epoch: {}'.format(_epoch))
    _model.eval()
    paths, labels, annotations, annotations_p_detections, num_frames = val_make_dataset()
    y_true = []
    pred_scores = []
    TUBE_BUILD_CONFIG['dataset_root'] = _data_root#'/media/david/datos/Violence DATA/UCFCrime2LocalClips/UCFCrime2LocalClips'

    for i, (path, label, annotation, annotation_p_detections, n_frames) in enumerate(zip(paths, labels, annotations, annotations_p_detections, num_frames)):
        print('\nprocession video: ', i+1)
        video_dataset = UCFCrime2LocalVideoDataset(cfg,
                                                    path=path,
                                                    sp_annotation=annotation,
                                                    clip_len=n_frames, #all frames
                                                    clip_temporal_stride=1,
                                                    transformations=transformations
                                                )
        # frames_names = os.listdir(path)
        # frames_names = natural_sort(frames_names)
        # frames_indices = list(range(0,len(frames_names)))
        person_detections = JSON_2_videoDetections(annotation_p_detections) #load person detections
        TUBE_BUILD_CONFIG['person_detections'] = person_detections
        for clip, frames_name, gt, num_frames in video_dataset:
            frames_indices = list(range(0,len(frames_name)))
            live_paths, time = extract_tubes_from_video(frames_indices, frames_name, MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG)
            print('num tubes: ', len(live_paths))
            boxes, video_images, labels, num_tubes, paths, key_frames = video_dataset.get_tube_data(tubes_=live_paths)
            print('video_images: ', video_images.size())
            print('key_frames: ', key_frames.size())
            print('boxes: ', boxes,  boxes.size())
            print('labels: ', labels)
            tube_scores = get_tube_scores(_model, video_images, key_frames, boxes, _device)
            print('tube_scores: ', tube_scores)
            max_idx = max_tube_idx(tube_scores)
            print('max_idx: ', max_idx)
            iou = st_iou(live_paths[max_idx], gt)
            print('iou: ', iou)