import add_libs
import os

import numpy as np
import torch
from configs.defaults import get_cfg_defaults
from configs.tube_config import MOTION_SEGMENTATION_CONFIG, TUBE_BUILD_CONFIG
from datasets.collate_fn import my_collate, my_collate_video
from datasets.dataloaders import two_stream_transforms
from models.TwoStreamVD_Binary_CFam import (TwoStreamVD_Binary_CFam,
                                            TwoStreamVD_Binary_CFam_Eval)

from torch.utils.data import DataLoader
from utils.global_var import *
from datasets.onevideo_dataset import VideoDemo
from utils.utils import get_torch_device, load_checkpoint

def score_tubes_by_motion(h_path):
    from datasets.make_dataset_handler import load_make_dataset
    from models.resnet import ResNet
    from utils.tube_utils import tube_2_JSON
    
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(WORK_DIR / "configs/DI_MODEL.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    cfg.DATA.DATASET = 'RWF-2000'
    cfg.DATA.CV_SPLIT = 1
    cfg.DATA.LOAD_GROUND_TRUTH = False
    
    cfg.TUBE_DATASET.NUM_FRAMES = 16
    # cfg.TUBE_DATASET.NUM_TUBES = 3
    cfg.TUBE_DATASET.RANDOM = False
    cfg.TUBE_DATASET.FRAMES_STRATEGY =   0
    cfg.TUBE_DATASET.BOX_STRATEGY = 1 #0:middle, 1: union, 2: all
    cfg.TUBE_DATASET.KEYFRAME_STRATEGY = 3 #0: rgb middle frame , 3:dynamic_images
    cfg.TUBE_DATASET.KEYFRAME_CROP = True
    cfg.MODEL.INFERENCE.CHECKPOINT_PATH = r'C:\Users\David\Desktop\DATASETS\Pretrained_Models\DI_MODEL_RESTNET_save_at_epoch-58.chk'
    
    make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        min_clip_len=-1,
                                        train=True,
                                        category=2,
                                        shuffle=False)
    make_dataset_val = load_make_dataset(cfg.DATA,
                                    env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                    min_clip_len=-1,
                                    train=False,
                                    category=2,
                                    shuffle=False)
    
    # paths, labels, annotations = make_dataset_train()
    paths, labels, annotations = make_dataset_val()
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    _device = get_torch_device()
    model = ResNet().to(_device)
    model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    model.eval()
    for path, label, annotation in zip(paths, labels, annotations):
        print('\n', path, '\n', label, '\n', annotation)
        
        
        out_annotation_scored = Path(annotation)
        file = out_annotation_scored.name
        clase = out_annotation_scored.parents[0].name
        split = out_annotation_scored.parents[1].name
        dataset = out_annotation_scored.parents[2].name
        folder = out_annotation_scored.parents[4]/Path('ActionTubesV2Scored')
        out_annotation_scored = folder/dataset/split/clase/file
        
        # print('Out file: ', out_annotation_scored)
        
        vd = VideoDemo(cfg=cfg.TUBE_DATASET,
                        path=path,
                        tub_file=annotation,
                        tub_cfg=TUBE_BUILD_CONFIG,
                        mot_cgf=MOTION_SEGMENTATION_CONFIG,
                        ped_file=None,
                        vizualize_tubes=True,
                        vizualize_keyframe=False,
                        transformations=transforms_config_val
                        )
        scored_tubes = []
        for tube_data in vd:
            f_box, tube_images_t, key_frame, tube = tube_data
            key_frame = torch.unsqueeze(key_frame, dim=0).to(_device)
            # print('keyframe: ', key_frame.size())
            # print('\ntube score:', tube['score'])
            pred = model(key_frame)
            # print('pred: ', pred, pred.size())
            max_scores, predicted = torch.max(pred, 1)
            # print('max_scores: ', max_scores)
            # print('label: ', predicted)
            
            # probs = torch.sigmoid(pred)
            # print("probs: ", probs, probs.size())
            
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(pred)
            # print("probs softmax: ", probabilities, probabilities.size())
            
            score = probabilities.detach().cpu().numpy()[0,1]
            tube['score'] = str(score)
            # print('score: ', score, tube['score'])
            scored_tubes.append(tube)
        if not os.path.isdir(out_annotation_scored.parents[0]): #Create folder of split
            os.makedirs(out_annotation_scored.parents[0])
        
        tube_2_JSON(out_annotation_scored, scored_tubes)
   
def violence_localization(h_path):
    from datasets.make_dataset_handler import load_make_dataset
    # from models.resnet import ResNet
    from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam
    from utils.tube_utils import tube_2_JSON
    
    cfg = get_cfg_defaults()
    # cfg.merge_from_file(WORK_DIR / "configs/DI_MODEL.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    cfg.DATA.DATASET = 'RWF-2000'
    cfg.DATA.CV_SPLIT = 1
    cfg.DATA.LOAD_GROUND_TRUTH = False
    
    cfg.TUBE_DATASET.NUM_FRAMES = 16
    # cfg.TUBE_DATASET.NUM_TUBES = 3
    cfg.TUBE_DATASET.RANDOM = False
    cfg.TUBE_DATASET.FRAMES_STRATEGY =   0
    cfg.TUBE_DATASET.BOX_STRATEGY = 1 #0:middle, 1: union, 2: all
    cfg.TUBE_DATASET.KEYFRAME_STRATEGY = 3 #0: rgb middle frame , 3:dynamic_images
    cfg.TUBE_DATASET.KEYFRAME_CROP = True
    cfg.MODEL.INFERENCE.CHECKPOINT_PATH = r'C:\Users\David\Desktop\DATASETS\Pretrained_Models\TWOSTREAM+I3Dv1+ResNet50+CFAM+TubesScored-RWF-save_at_epoch-35.chk'
    
    make_dataset_val = load_make_dataset(cfg.DATA,
                                    env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                    min_clip_len=-1,
                                    train=False,
                                    category=2,
                                    shuffle=False)
    
    # paths, labels, annotations = make_dataset_train()
    paths, labels, annotations = make_dataset_val()
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    _device = get_torch_device()
    model = TwoStreamVD_Binary_CFam(cfg).to(_device)
    model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    model.eval()
    for path, label, annotation in zip(paths, labels, annotations):
        print('\n', path, '\n', label, '\n', annotation)
        
        out_annotation_scored = Path(annotation)
        file = out_annotation_scored.name
        clase = out_annotation_scored.parents[0].name
        split = out_annotation_scored.parents[1].name
        dataset = out_annotation_scored.parents[2].name
        folder = out_annotation_scored.parents[4]/Path('ActionTubesV2Scored')
        out_annotation_scored = folder/dataset/split/clase/file
        
        # print('Out file: ', out_annotation_scored)
        
        vd = VideoDemo(cfg=cfg.TUBE_DATASET,
                        path=path,
                        tub_file=annotation,
                        tub_cfg=TUBE_BUILD_CONFIG,
                        mot_cgf=MOTION_SEGMENTATION_CONFIG,
                        ped_file=None,
                        vizualize_tubes=True,
                        vizualize_keyframe=False,
                        transformations=transforms_config_val
                        )
        scored_tubes = []
        for tube_data in vd:
            f_box, tube_images_t, key_frame, tube = tube_data
            key_frame = torch.unsqueeze(key_frame, dim=0).to(_device)
            # print('keyframe: ', key_frame.size())
            # print('\ntube score:', tube['score'])
            pred = model(key_frame)
            # print('pred: ', pred, pred.size())
            max_scores, predicted = torch.max(pred, 1)
            # print('max_scores: ', max_scores)
            # print('label: ', predicted)
            
            # probs = torch.sigmoid(pred)
            # print("probs: ", probs, probs.size())
            
            sm = torch.nn.Softmax(dim=1)
            probabilities = sm(pred)
            # print("probs softmax: ", probabilities, probabilities.size())
            
            score = probabilities.detach().cpu().numpy()[0,1]
            tube['score'] = str(score)
            # print('score: ', score, tube['score'])
            scored_tubes.append(tube)
        if not os.path.isdir(out_annotation_scored.parents[0]): #Create folder of split
            os.makedirs(out_annotation_scored.parents[0])
        
        tube_2_JSON(out_annotation_scored, scored_tubes)
    
    
def demo():
    # rwf_config = {
    #     'dataset_root': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RWF-2000/frames',
    #     'split': 'train/Fight',
    #     'video': 'u1r8f71c_3',#'dt8YUGoOSgQ_0',#'C8wt47cphU8_1',#'_2RYnSFPD_U_0',
    #     'p_d_path': '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/RWF-2000'
    # }
    # config = rwf_config
    # persons_detections_path = config['p_d_path']+'/{}/{}.json'.format(config['split'],config['video'])
    # person_detections = JSON_2_videoDetections(persons_detections_path)
    # frames = np.linspace(0, 149,dtype=np.int16).tolist()
    # # frames = np.linspace(0, 149, dtype=np.int16).tolist()
    # print('random frames: ', frames)

    # TUBE_BUILD_CONFIG['dataset_root'] = config['dataset_root']
    # TUBE_BUILD_CONFIG['person_detections'] = person_detections

    # # plot_create_save_dirs(config)
    # TUBE_BUILD_CONFIG['plot_config']['plot_tubes'] = True
    # TUBE_BUILD_CONFIG['plot_config']['debug_mode'] = True

    # live_paths = extract_tubes_from_video(
    #     frames
    #     )
    
    # print('live_paths: ', len(live_paths))
    # print(live_paths[0])

    h_path = HOME_WINDOWS
    cfg = get_cfg_defaults()
    cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_3DRoiPool_2DRoiPool.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    cfg.MODEL.INFERENCE.CHECKPOINT_PATH = r'C:\Users\David\Desktop\DATASETS\Pretrained_Models\TWOSTREAM+I3Dv1+ResNet50+CFAM+TubesScored-RWF-save_at_epoch-35.chk'
    # _6-B11R9FJM_2 (TP)
    # 0_DzLlklZa0_5 (TN)
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    _device = get_torch_device()
    model = TwoStreamVD_Binary_CFam(cfg.MODEL).to(_device)
    model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    # model = TwoStreamVD_Binary_CFam_Eval(model)
    # print(model)
    model.eval()
    
    
        
    def iterate(category, set_, video):
        vd = VideoDemo(
            cfg=cfg.TUBE_DATASET,
            path=r"C:\Users\David\Desktop\DATASETS\RWF-2000\frames\{}\{}\{}".format(set_, category, video),
            tub_file=r"C:\Users\David\Desktop\DATASETS\ActionTubesV2Scored\RWF-2000\{}\{}\{}.json".format(set_, category, video),
            tub_cfg=TUBE_BUILD_CONFIG,
            mot_cgf=MOTION_SEGMENTATION_CONFIG,
            ped_file=r"C:\Users\David\Desktop\DATASETS\PersonDetections\RWF-2000\{}\{}\{}.json".format(set_, category, video),
            vizualize_tubes=False,
            save_folder=r"C:\Users\David\Desktop\DATASETS\Vizualizations",
            # vizualize_keyframe=True,
            transformations=transforms_config_val
        )
        
        loader = DataLoader(vd,
                            batch_size=1,
                            # shuffle=False,
                            num_workers=1,
                            # pin_memory=True,
                            collate_fn=my_collate_video,
                            # sampler=get_sampler(train_dataset.labels),
                            drop_last=False
                            )
        
        
        tube_scores = []
        for batch_idx, (box, tube_images, keyframe) in enumerate(loader):
            print("\nbatch: ", batch_idx)
            
            print("box: ", box.size(), '\n', box)
            print("tube_images: ", tube_images.size())
            print("keyframe: ", keyframe.size())
            
            box, tube_images = box.to(_device), tube_images.to(_device)
            keyframe = keyframe.to(_device)
            with torch.no_grad():
                outputs = model(tube_images, keyframe, box, cfg.TUBE_DATASET.NUM_TUBES)
                outputs = outputs.unsqueeze(dim=0) #tensor([[0.0735, 0.1003]]) torch.Size([b, 2])
                print('outputs: ', outputs)
                max_scores, predicted = torch.max(outputs, 1)
                print('max_scores: ', max_scores)
                print('predicted: ', predicted)
                
                probs = torch.sigmoid(outputs)
                print("probs: ", probs, probs.size())
                
                sm = torch.nn.Softmax(dim=1)
                probabilities = sm(outputs)
                print("probs softmax: ", probabilities, probabilities.size())
                
                tube_scores.append(probabilities)
        
        tube_scores = torch.cat(tube_scores, dim=0)
        print("tube_scores: ", tube_scores, tube_scores.size())
        
        if len(vd) >= 1 or len(vd) <= 3:
            t_max_scores, indices = torch.max(tube_scores, 0)
            indice_max_tube = indices.cpu().numpy().tolist()[1]
            vd.plot_best_tube([indice_max_tube])
        elif len(vd) > 3:
            t_max_scores, indices = torch.topk(tube_scores, 3, dim=0)
            print('t_max_scores: ', t_max_scores)
            print('indices: ', indices)
            indice_max_tube = indices[:,1].cpu().numpy().tolist()
            print('indice_max_tube: ', indice_max_tube)
            vd.plot_best_tube(indice_max_tube)

    set_ = 'val'
    category = 'NonFight'
    videos = ['1AURh0Wj_0', '2qFnVnnZ_0', '4I1DGWsh_0', '2lrARl7utL4_0', '1W8hsVvyKt4_1', '6doZoiaG9PM_1', '7emxm3za_0', '8SmT9rE9_0', '66U2c3YMjOI_0', 'A7FCl8G35Cs_0', 'bW2vHhYbzHM_0']
    # set_ = 'train'
    # video = '0_DzLlklZa0_5'
    for video in videos:
        iterate(category, set_, video)
        
if __name__=='__main__':
    demo()
    
    # h_path = HOME_WINDOWS
    # score_tubes(h_path)
