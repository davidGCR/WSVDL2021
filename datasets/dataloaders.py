from utils.global_var import *

from datasets.tube_dataset import TubeDataset
from datasets.collate_fn import my_collate
from datasets.cnn_input_config import CnnInputConfig
from transformations.model_transforms import *

from torch.utils.data import DataLoader

def load_key_frame_config(cfg, split):
    """Build config dict for input of 2d_Branch

    Args:
        cfg (yaml): config file of TubeDataset
        split (str): split train or val

    Returns:
        CnnInputConfig: object with config of the 2d branch
    """
    if cfg.KEYFRAME_STRATEGY in [RGB_BEGIN_KEYFRAME, RGB_MIDDLE_KEYFRAME, RGB_RANDOM_KEYFRAME]:
        # input_2_c={
        #         'type': 'rgb',
        #         'spatial_transform': resnet_transf()[split],
        #         'temporal_transform': None
        #     }
        input_2_c = CnnInputConfig()
        input_2_c.itype = RGB_FRAME
        input_2_c.spatial_transform = resnet_transf()[split]

    elif cfg.KEYFRAME_STRATEGY == DYNAMIC_IMAGE_KEYFRAME:
        # input_2_c = {
        #     'type': 'dynamic-image',
        #     'spatial_transform': resnet_di_transf()[split],
        #     'temporal_transform': None
        # }
        input_2_c = CnnInputConfig()
        input_2_c.itype = DYN_IMAGE
        input_2_c.spatial_transform = resnet_di_transf()[split]
    else:
        print('Error loading key_frame_config. No valid keyframe...')
        exit()
    return input_2_c

def data_with_tubes(cfg, make_dataset_train, make_dataset_val):
    """Build dataloaders for train and val sets.

    Args:
        cfg (yaml): Main yaml file
        make_dataset_train (function): make function of train set
        make_dataset_val (function): make function of val/test set

    Returns:
        tuple: (train, val) dataloaders
    """
    # TWO_STREAM_INPUT_train = {
    #     'input_1': {
    #         'type': 'rgb',
    #         'spatial_transform': cnn3d_transf()['train'],
    #         'temporal_transform': None
    #     },
    #     'input_2': load_key_frame_config(cfg.TUBE_DATASET, 'train') 
    # }
    TWO_STREAM_INPUT_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET, 'train') 
    }
    
    # TWO_STREAM_INPUT_val = {
    #     'input_1': {
    #         'type': 'rgb',
    #         # 'spatial_transform': i3d_video_transf()['val'],
    #         'spatial_transform': cnn3d_transf()['val'],
    #         'temporal_transform': None
    #     },
    #     'input_2': load_key_frame_config(cfg.TUBE_DATASET, 'val')
    # }
    TWO_STREAM_INPUT_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET, 'val') 
    }

    train_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_train, TWO_STREAM_INPUT_train, cfg.DATA.DATASET)
    train_loader = DataLoader(train_dataset,
                        batch_size=cfg.DATALOADER.TRAIN_BATCH,
                        # shuffle=False,
                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                        # pin_memory=True,
                        collate_fn=my_collate,
                        sampler=train_dataset.get_sampler(),
                        drop_last=cfg.DATALOADER.DROP_LAST
                        )
    val_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_val, TWO_STREAM_INPUT_val, cfg.DATA.DATASET)
    val_loader = DataLoader(val_dataset,
                        batch_size=cfg.DATALOADER.VAL_BATCH,
                        # shuffle=True,
                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                        sampler=val_dataset.get_sampler(),
                        # pin_memory=True,
                        collate_fn=my_collate,
                        drop_last=cfg.DATALOADER.DROP_LAST
                        )
    return train_loader, val_loader



def data_with_tubes_localization(cfg, make_dataset_train):
    """Build train dataloader for train dataset and CnnInputConfig for val set.

    Args:
        cfg (yaml): Main yaml file
        make_dataset_train (function): make function of train set
        make_dataset_val (function): make function of val/test set

    Returns:
        tuple: (train_loader, dict{input_1: CnnInputConfig, input_2: CnnInputConfig})
    """

    TWO_STREAM_INPUT_train = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['train'], None),
        'input_2': load_key_frame_config(cfg.TUBE_DATASET, 'train') 
    }

    TWO_STREAM_INPUT_val = {
        'input_1': CnnInputConfig(RGB_FRAME, cnn3d_transf()['val'], None), 
        'input_2': load_key_frame_config(cfg.TUBE_DATASET, 'val')
    }
    
    train_dataset = TubeDataset(cfg.TUBE_DATASET, make_dataset_train, TWO_STREAM_INPUT_train, cfg.DATA.DATASET)
    train_loader = DataLoader(train_dataset,
                        batch_size=cfg.DATALOADER.TRAIN_BATCH,
                        # shuffle=False,
                        num_workers=cfg.DATALOADER.NUM_WORKERS,
                        # pin_memory=True,
                        collate_fn=my_collate,
                        sampler=train_dataset.get_sampler(),
                        drop_last=cfg.DATALOADER.DROP_LAST
                        )
    return train_loader, TWO_STREAM_INPUT_val