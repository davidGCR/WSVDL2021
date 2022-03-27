from msilib import sequence
import add_libs
import os
import numpy as np
from utils.tube_utils import JSON_2_videoDetections, JSON_2_tube
from utils.utils import natural_sort
from tubes.run_tube_gen import extract_tubes_from_video
from tubes.plot_tubes import plot_tubes
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from pathlib import Path
from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam, TwoStreamVD_Binary_CFam_Eval
from configs.defaults import get_cfg_defaults
from utils.global_var import *
from utils.utils import get_torch_device, load_checkpoint
from utils.dataset_utils import imread
from datasets.dataloaders import two_stream_transforms
from transformations.dynamic_image_transformation import DynamicImage
import torch.utils.data as data
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.collate_fn import my_collate, my_collate_video

def build_frame_name(path, frame_number, frames_names_list):
    frame_idx = frame_number
    pth = os.path.join(path, frames_names_list[frame_idx])
    return pth

def __format_bbox__(bbox, shape, box_as_tensor=False):
    """
    Format a tube bbox: [x1,y1,x2,y2] to a correct format
    """
    (width, height) = shape
    bbox = bbox[0:4]
    bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
    # bbox = np.insert(bbox[0:4], 0, id).reshape(1,-1).astype(float)
    bbox = bbox.reshape(1,-1).astype(float)
    if box_as_tensor:
        bbox = torch.from_numpy(bbox).float()
    return bbox
    
def load_input_1(path:str, frames_indices:list, frames_names_list:list, sampled_tube:dict, transformations:dict, shape:tuple):
    """Gets tensors from an action tube

    Args:
        path (str): Path to the video folder
        frames_indices (list): _description_
        frames_names_list (list): _description_
        sampled_tube (dict): _description_
        transformations (dict): _description_
        shape (tuple): _description_

    Returns:
        tuple: tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images, t_combination
    """
    # print('\nload_input_1--> frames_paths')
    tube_images = []
    raw_clip_images = []
    tube_images_t = None
    tube_boxes = []
    if transformations['input_1'].itype=='rgb':
        frames_paths = [build_frame_name(path, i, frames_names_list) for i in frames_indices]
        # for j, fp in enumerate(frames_paths):
        #     print(j, ' ', fp)
        for i in frames_paths:
            img = imread(i)
            tube_images.append(img)
            _, frame_name = os.path.split(i)
            
            try:
                box_idx = sampled_tube['frames_name'].index(frame_name)
            except Exception as e:
                print("\nOops!", e.__class__, "occurred.")
                print("sampled_tube['frames_name']: {}, frame: {} , sampled_indices: {}, path: {}".format(sampled_tube['frames_name'], frame_name, frames_indices, path))
                exit()
            tube_boxes.append(box_idx)
        
        tube_boxes = [sampled_tube['boxes'][b] for b in tube_boxes]
        tube_boxes = [__format_bbox__(t, shape) for t in tube_boxes]
        
        # print('\tube_boxes: ', tube_boxes, len(tube_boxes))
        # print('\t tube_images: ', type(tube_images), type(tube_images[0]))
        raw_clip_images = tube_images.copy()
        if transformations['input_1'].spatial_transform:
            tube_images_t, tube_boxes_t, t_combination = transformations['input_1'].spatial_transform(tube_images, tube_boxes)
    
    return tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images, t_combination
    
def get_tube_data(tube: dict, 
                  tube_frames_indices: list, 
                  frames_names_list: list, 
                  video_path: str, 
                  box_trategy: str,
                  transformations: dict,
                  shape: tuple,
                  keyframe_strategy: int,
                  dyn_fn: any):
    """Gets tensors from an action tube

    Args:
        tube (dict): Action tube
        tube_frames_indices (list): List of integers corresponding to indices in a list of frames
        frames_names_list (list): List of imgs files.
        video_path (str): Path to the video folder with frames
        box_trategy (str): Type of tube box sampling.
        transformations (dict): Spatial transformations for two stream model.
        shape (tuple): Spatial size.
        keyframe_strategy (int): Type of keyframe from tube,
        dyn_fn (function): Function to compute a dynamic image
    """
    input_1 = load_input_1(video_path, tube_frames_indices, frames_names_list, tube, transformations, shape)
    tube_images_t, tube_boxes_t, tube_boxes, tube_raw_clip_images, t_combination = input_1
    tube_images_t = torch.stack(tube_images_t, dim=0)
    
    #Box extracted from tube
    tube_box = None
    if box_trategy == MIDDLE_BOX:
        m = int(len(tube_boxes)/2) #middle box from tube
        ##setting id to box
        tube_box = tube_boxes_t[m]
        id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
        # print('\n', ' id_tensor: ', id_tensor,id_tensor.size())
        # print(' c_box: ', c_box, c_box.size(), ' index: ', m)
        if tube_box.size(0)==0:
            print('get_tube_data error in tube_box: ', video_path, '\n',
                    tube_box, '\n', 
                    tube, '\n', 
                    tube_frames_indices, '\n', 
                    tube_boxes_t, len(tube_boxes_t), '\n', 
                    tube_boxes, len(tube_boxes), '\n',
                    t_combination)
            exit()
        f_box = torch.cat([id_tensor , tube_box], dim=1).float()
    elif box_trategy == UNION_BOX:
        all_boxes = [torch.from_numpy(t) for i, t in enumerate(tube_boxes)]
        all_boxes = torch.stack(all_boxes, dim=0).squeeze()
        mins, _ = torch.min(all_boxes, dim=0)
        x1 = mins[0].unsqueeze(dim=0).float()
        y1 = mins[1].unsqueeze(dim=0).float()
        maxs, _ = torch.max(all_boxes, dim=0)
        x2 = maxs[2].unsqueeze(dim=0).float()
        y2 = maxs[3].unsqueeze(dim=0).float()
        id_tensor = torch.tensor([0]).float()
        
        f_box = torch.cat([id_tensor , x1, y1, x2, y2]).float()
    elif box_trategy == ALL_BOX:
        f_box = [torch.cat([torch.tensor([i]).unsqueeze(dim=0), torch.from_numpy(t)], dim=1).float() for i, t in enumerate(tube_boxes)]
        f_box = torch.stack(f_box, dim=0)
    
    f_box = torch.unsqueeze(f_box, dim=0)
    # print('tube_box: ', f_box.size())
    
    #load keyframes
    # key_frames = []
    if transformations['input_2'] is not None:
        if keyframe_strategy == DYNAMIC_IMAGE_KEYFRAME:
            # key_frame, _ = self.load_input_2_di(sampled_frames_indices[k], path, frames_names_list)
            key_frame = dyn_fn(tube_images_t)
            if transformations['input_2'].spatial_transform:
                key_frame = transformations['input_2'].spatial_transform(key_frame)
        else:
            if keyframe_strategy == RGB_MIDDLE_KEYFRAME:
                m = int(tube_images_t.size(0)/2) #using frames loaded from 3d branch
                key_frame = tube_images_t[m] #tensor 
                key_frame = key_frame.numpy()
                if transformations['input_2'].spatial_transform:
                    key_frame = transformations['input_2'].spatial_transform(key_frame)
            else:
                #TODO
                print('Not implemented yet...')
                exit()
    # print('keyframe: ', key_frame.size())
    tube_images_t = tube_images_t.permute(3,0,1,2)#.permute(0,2,1,3,4)
    # print('tube_images_t: ', tube_images_t.size())
    return f_box, tube_images_t, key_frame

class VideoDemo(data.Dataset):
    def __init__(self, 
                 cfg,
                 path: str, 
                 tub_cfg: dict, 
                 mot_cgf: dict, 
                 tub_file=None, 
                 ped_file=None, 
                 vizualize_tubes:bool=False,
                 transformations=None):
        """Dataset to load tubes from a single video

        Args:
            cfg (Yaml): cfg.TUBE_DATASET.
            path (str): Path to video folder with frames.
            tub_cfg (dict): Configuration for online tube generation. Obligatory to vizualization.
            mot_cgf (dict): Configuration for online motion segmentation.
            tub_file (_type_, optional): Path to json file with precomputed tubes. Defaults to None.
            ped_file (_type_, optional): Path to json file with person detections. Defaults to None.
            vizualize_tubes (bool, optional): Flac to vizualize online tube generation. Defaults to False.
            transformations (dict, optional): Spatial transformations for two stream model. Defaults to False.
        """
        self.cfg = cfg
        self.path = Path(path)
        self.check_file(self.path)
        self.vizualize_tubes = vizualize_tubes
        
        self.tub_file = self.check_file(tub_file)
        self.ped_file = self.check_file(ped_file)
        # self.frames = natural_sort([frame for frame in os.listdir(self.path) if '.jpg' in frame])
        self.frames = natural_sort([p.name for p in self.path.iterdir()])
        self.num_frames = len(self.frames)
        
        #path root
        self.root = self.path.parents[2]

        # Tube generation config
        self.tub_cfg = tub_cfg
        self.tub_cfg['dataset_root'] = self.root
        self.mot_cgf = mot_cgf
        self.transformations = transformations
        
        if vizualize_tubes:
            self.tub_cfg['plot_config']['plot_tubes'] = True
            self.tub_cfg['plot_config']['debug_mode'] = False

        self.tubes = self.gen_tubes()
        print('num tubes: ', len(self.tubes))
    
    def __len__(self):
        return len(self.tubes)
    
    def __getitem__(self, index):
        tube = self.tubes[index]
        tube_frames_idxs = self.__sampled_tube_frames_indices__(tube['foundAt'], self.cfg.NUM_FRAMES, self.cfg.FRAMES_STRATEGY)
        f_box, tube_images_t, key_frame = get_tube_data(tube, 
                                                        tube_frames_idxs, 
                                                        self.frames, 
                                                        self.path, 
                                                        self.cfg.BOX_STRATEGY, 
                                                        self.transformations, 
                                                        self.cfg.SHAPE,
                                                        self.cfg.KEYFRAME_STRATEGY,
                                                        DynamicImage())
        return f_box, tube_images_t, key_frame
    
    def __sampled_tube_frames_indices__(self, 
                                        tube_found_at: list,
                                        tube_len,
                                        sample_strategy):
        max_video_len = tube_found_at[-1]
        if len(tube_found_at) == tube_len: 
            return tube_found_at
        if len(tube_found_at) > tube_len:
            if sample_strategy == MIDDLE_FRAMES:
                n = len(tube_found_at)
                m = int(n/2)
                arr = np.array(tube_found_at)
                centered_array = arr[m-int(tube_len/2) : m+int(tube_len/2)]
            elif sample_strategy == EVENLY_FRAMES:
                min_frame = tube_found_at[0]
                tube_frames_idxs = np.linspace(min_frame, max_video_len, tube_len).astype(int)
                tube_frames_idxs = tube_frames_idxs.tolist()
            return centered_array.tolist()
        if len(tube_found_at) < tube_len: #padding
            min_frame = tube_found_at[0]
            # TODO 
            tube_frames_idxs = np.linspace(min_frame, max_video_len, tube_len).astype(int)
            tube_frames_idxs = tube_frames_idxs.tolist()
            return tube_frames_idxs
        
    def check_file(self, path):
        if path:
            path = Path(path)
            if not path.exists():
                print('ERROR: File: {} does not exist!!!'.format(path))
        return path

    def temporal_step(self):
        # indices = np.linspace(0, self.num_frames, dtype=np.int16).tolist()
        indices = list(range(0, self.num_frames))
        names = [str(self.path/self.frames[i]) for i in indices]
        return indices, names
    
    def plot_best_tube(self, index):
        tubes = JSON_2_tube(str(self.tub_file))
        indices, names = self.temporal_step()
        plot_tubes(names, [tubes[index]])
        
    def gen_tubes(self):
        tubes = None
        if self.tub_file:
            print('Loading tubes from: ', self.tub_file)
            tubes = JSON_2_tube(str(self.tub_file))
            indices, names = self.temporal_step()
            if self.vizualize_tubes:
                plot_tubes(names, tubes)
        else:
            print('Extracting tubes...')
            if self.ped_file:
                person_detections = JSON_2_videoDetections(str(self.ped_file))
                self.tub_cfg['person_detections'] = person_detections
                indices, names = self.temporal_step()
                tubes, time = extract_tubes_from_video(indices, names, self.mot_cgf, self.tub_cfg, None)
            else:
                print('ERROR: No persons detections file!!!')
        
        return tubes

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
    # _6-B11R9FJM_2 (TP)
    # 0_DzLlklZa0_5 (TN)
    transforms_config_train, transforms_config_val = two_stream_transforms(cfg.TUBE_DATASET.KEYFRAME_STRATEGY)
    # set_ = 'val'
    # video = '1Kbw1bUw_1'
    set_ = 'train'
    video = '0_DzLlklZa0_5'
    vd = VideoDemo(
        cfg=cfg.TUBE_DATASET,
        path=r"C:\Users\David\Desktop\DATASETS\RWF-2000\frames\{}\Fight\{}".format(set_, video),
        tub_file=r"C:\Users\David\Desktop\DATASETS\ActionTubesV2\RWF-2000\{}\Fight\{}.json".format(set_, video),
        tub_cfg=TUBE_BUILD_CONFIG,
        mot_cgf=MOTION_SEGMENTATION_CONFIG,
        ped_file=r"C:\Users\David\Desktop\DATASETS\PersonDetections\RWF-2000\{}\Fight\{}.json".format(set_, video),
        vizualize_tubes=True,
        transformations=transforms_config_val
    )
    
    # loader = DataLoader(vd,
    #                     batch_size=4,
    #                     # shuffle=False,
    #                     num_workers=1,
    #                     # pin_memory=True,
    #                     collate_fn=my_collate_video,
    #                     # sampler=get_sampler(train_dataset.labels),
    #                     drop_last=False
    #                     )

    # # for i in range(len(vd)):
    # #     print('\ntube: {}'.format(i+1))
    # #     tb = vd[i]
        
    # #     print(tb)
    
    # _device = get_torch_device()
    # model = TwoStreamVD_Binary_CFam(cfg.MODEL).to(_device)
    # model, _, _, _, _ = load_checkpoint(model, _device, None, cfg.MODEL.INFERENCE.CHECKPOINT_PATH)
    # # model = TwoStreamVD_Binary_CFam_Eval(model)
    # # print(model)
    # model.eval()
    # tube_scores = []
    # for batch_idx, (box, tube_images, keyframe) in enumerate(loader):
    #     print("\nbatch: ", batch_idx)
        
    #     # print("box: ", box.size(), '\n', box)
    #     # print("tube_images: ", tube_images.size())
    #     # print("keyframe: ", keyframe.size())
        
    #     box, tube_images = box.to(_device), tube_images.to(_device)
    #     keyframe = keyframe.to(_device)
    #     with torch.no_grad():
    #         outputs = model(tube_images, keyframe, box, cfg.TUBE_DATASET.NUM_TUBES)
    #         outputs = outputs.unsqueeze(dim=0) #tensor([[0.0735, 0.1003]]) torch.Size([b, 2])
    #         max_scores, predicted = torch.max(outputs, 1)
    #         print('max_scores: ', max_scores)
    #         print('predicted: ', predicted)
            
    #         probs = torch.sigmoid(outputs)
    #         print("probs: ", probs, probs.size())
            
    #         sm = torch.nn.Softmax(dim=1)
    #         probabilities = sm(outputs)
    #         print("probs softmax: ", probabilities, probabilities.size())
            
    #         tube_scores.append(probabilities)
    
    # tube_scores = torch.cat(tube_scores, dim=0)
    # print("tube_scores: ", tube_scores, tube_scores.size())
    # t_max_scores, indices = torch.max(tube_scores, 0)
    # print('t_max_scores: ', t_max_scores)
    # print('indices: ', indices)
    # indice_max_tube = indices.cpu().numpy().tolist()[1]
    # print('indice_max_tube: ', indice_max_tube)
    # vd.plot_best_tube(indice_max_tube)
    
        
        
        
        
    

if __name__=='__main__':
    demo()