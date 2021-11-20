



# import imports
import os
import random
import numpy as np
from numpy.core.numeric import indices
import json

import torch
import torch.utils.data as data
from torch.utils.data import dataset
from torch.utils.data.sampler import WeightedRandomSampler

# from datasets.make_dataset import *
# from datasets.make_UCFCrime import *
from datasets.tube_crop import TubeCrop

from transformations.dynamic_image_transformation import DynamicImage

from utils.global_var import *
from utils.utils import natural_sort
from utils.dataset_utils import imread, filter_data_without_tubelet, JSON_2_tube, check_no_tubes

class TubeDataset(data.Dataset):
    def __init__(self, cfg, make_fn, inputs_config, dataset):
        """Init a TubeDataset

        Args:
            cfg (Yaml): cfg.TUBE_DATASET
            make_fn ([type]): [description]
            inputs_config ([type]): [description]
            dataset ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.cfg = cfg
        self.make_function = make_fn
        self.config = inputs_config
        self.dataset = dataset
        if self.dataset == 'UCFCrime':
            self.paths, self.labels, _, self.annotations, self.num_frames = self.make_function()
            # indices_2_remove = []
            # for index in range(len(self.paths)):
            #     annotation = self.annotations[index]
            #     if len(annotation) == 0:
            #         indices_2_remove.append(index)
            # self.paths = [self.paths[i] for i in range(len(self.paths)) if i not in indices_2_remove]
            # self.labels = [self.labels[i] for i in range(len(self.labels)) if i not in indices_2_remove]
            # self.annotations = [self.annotations[i] for i in range(len(self.annotations)) if i not in indices_2_remove]
        elif self.dataset == UCFCrimeReduced_DATASET:
            self.paths, self.labels, self.annotations, self.num_frames = self.make_function()
        else:
            self.paths, self.labels, self.annotations = self.make_function()
            self.paths, self.labels, self.annotations = filter_data_without_tubelet(self.paths, self.labels, self.annotations)


        print('paths: {}, labels:{}, annot:{}'.format(len(self.paths), len(self.labels), len(self.annotations)))
        self.sampler = TubeCrop(tube_len=cfg.NUM_FRAMES,
                                central_frame=True,
                                max_num_tubes=cfg.NUM_TUBES,
                                input_type=self.config['input_1']['type'],
                                sample_strategy=cfg.FRAMES_STRATEGY,
                                random=cfg.RANDOM,
                                box_as_tensor=False)

        if self.config['input_2']['type'] == DYN_IMAGE:
            self.dynamic_image_fn = DynamicImage()
    
    def get_sampler(self):
        class_sample_count = np.unique(self.labels, return_counts=True)[1]
        weight = 1./class_sample_count
        print('class_sample_count: ', class_sample_count)
        print('weight: ', weight)
        samples_weight = weight[self.labels]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler
    
    def build_frame_name(self, path, frame_number, frames_names_list):
        if self.dataset == RWF_DATASET:
            return os.path.join(path,'frame{}.jpg'.format(frame_number+1))
        elif self.dataset == HOCKEY_DATASET:
            return os.path.join(path,'frame{:03}.jpg'.format(frame_number+1))
        elif self.dataset == RLVSD_DATASET:
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
        elif self.dataset == UCFCrime_DATASET:
            return os.path.join(path,'{:06}.jpg'.format(frame_number))
        elif self.dataset == UCFCrimeReduced_DATASET:
            frame_idx = frame_number
            pth = os.path.join(path, frames_names_list[frame_idx])
            return pth
    
    def __format_bbox__(self, bbox):
        """
        Format a tube bbox: [x1,y1,x2,y2] to a correct format
        """
        (width, height) = self.cfg.SHAPE
        bbox = bbox[0:4]
        bbox = np.array([max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], width - 1), min(bbox[3], height - 1)])
        # bbox = np.insert(bbox[0:4], 0, id).reshape(1,-1).astype(float)
        bbox = bbox.reshape(1,-1).astype(float)
        if self.cfg.BOX_AS_TENSOR:
            bbox = torch.from_numpy(bbox).float()
        return bbox
    
    def load_input_1(self, path, frames_indices, frames_names_list, sampled_tube):
        # print('\nload_input_1--> frames_paths')
        tube_images = []
        raw_clip_images = []
        tube_images_t = None
        tube_boxes = []
        if self.config['input_1']['type']=='rgb':
            frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames_indices]
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
            tube_boxes = [self.__format_bbox__(t) for t in tube_boxes]
            
            # print('\tube_boxes: ', tube_boxes, len(tube_boxes))
            # print('\t tube_images: ', type(tube_images), type(tube_images[0]))
            raw_clip_images = tube_images.copy()
            if self.config['input_1']['spatial_transform']:
                tube_images_t, tube_boxes_t, t_combination = self.config['input_1']['spatial_transform'](tube_images, tube_boxes)
       
        return tube_images_t, tube_boxes_t, tube_boxes, raw_clip_images, t_combination
    
    def load_input_2_di(self, frames_indices, path, frames_names_list):
        # if self.config['input_2']['type'] == 'rgb':
        #     i = frames_indices[int(len(frames_indices)/2)]
            
        #     img_path = self.build_frame_name(path, i, frames_names_list)
        #     print('central frame path: ', i, ' ', img_path)
        #     key_frame = imread(img_path)
            
        # elif self.config['input_2']['type'] == 'dynamic-image':
        frames_paths = [self.build_frame_name(path, i, frames_names_list) for i in frames_indices] #rwf
        print('frames to build DI')
        for j, fp in enumerate(frames_paths):
            print(j, ' ', fp)
        shot_images = [np.array(imread(img_path, resize=self.shape)) for img_path in frames_paths]
        key_frame = self.dynamic_image_fn(shot_images)
            
        raw_key_frame = key_frame.copy()
        if self.config['input_2']['spatial_transform']:
            key_frame = self.config['input_2']['spatial_transform'](key_frame)
        return key_frame, raw_key_frame

    def load_tube_images(self, path, seg):
        tube_images = [] #one tube-16 frames
        if self.input_type=='rgb':
            frames = [self.build_frame_name(path, i) for i in seg]
            for i in frames:
                img = imread(i)
                tube_images.append(img)
        else:
            tt = DynamicImage()
            for shot in seg:
                if self.dataset == 'rwf-2000':
                    frames = [os.path.join(path,'frame{}.jpg'.format(i+1)) for i in shot] #rwf
                elif self.dataset == 'hockey':
                    frames = [os.path.join(path,'frame{:03}.jpg'.format(i+1)) for i in shot]
                shot_images = [imread(img_path) for img_path in frames]
                img = self.spatial_transform(tt(shot_images)) if self.spatial_transform else tt(shot_images)
                tube_images.append(img)
        return tube_images

    def load_tube_from_file(self, annotation):
        if self.dataset == 'UCFCrime':
            return annotation
        else:
            if isinstance(annotation, list):
                video_tubes = annotation
            else:
                video_tubes = JSON_2_tube(annotation)
            assert len(video_tubes) >= 1, "No tubes in video!!!==>{}".format(annotation)
            return video_tubes
    
    def video_max_len(self, idx):
        path = self.paths[idx]
        if self.dataset == 'RealLifeViolenceDataset':
            max_video_len = len(os.listdir(path)) - 1
        elif self.dataset=='hockey':
            max_video_len = 39
        elif self.dataset=='rwf-2000':
            max_video_len = 149
        elif self.dataset == 'UCFCrime':
            max_video_len = self.annotations[idx][0]['foundAt'][-1]- 1
        elif self.dataset == 'UCFCrime_Reduced':
            max_video_len = len(os.listdir(path)) - 1
        return max_video_len

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        frames_names_list = os.listdir(path)
        frames_names_list = natural_sort(frames_names_list)
        # print('frames_names_list: ', frames_names_list)
        
        label = self.labels[index]
        annotation = self.annotations[index]
        
        # max_video_len = self.video_max_len(index)
        tubes_ = self.load_tube_from_file(annotation)
        #remove tubes with len=1
        # tubes_ = [t for t in tubes_ if t['len'] > 1]
        # print('\n\ntubes_: ', tubes_)
        sampled_frames_indices, chosed_tubes = self.sampler(tubes_)

        # for i in range(len(sampled_frames_indices)):
        #     print('\ntube[{}] \n (1)frames_names_list: {}, \n(2)tube frames_name: {}, \n(3)sampled_frames_indices: {}'.format(i,frames_names_list, chosed_tubes[i]['frames_name'], sampled_frames_indices[i]))
        # print('sampled_frames_indices: ', sampled_frames_indices)
        # print('boxes_from_sampler: ', boxes, boxes[0].shape)
        video_images = []
        video_images_raw = []
        final_tube_boxes = []
        num_tubes = len(sampled_frames_indices)
        for frames_indices, sampled_tube in zip(sampled_frames_indices, chosed_tubes):
            # print('\nload_input_1 args: ', path, frames_indices, boxes)
            tube_images_t, tube_boxes_t, tube_boxes, tube_raw_clip_images, t_combination = self.load_input_1(path, frames_indices, frames_names_list, sampled_tube)
            video_images.append(torch.stack(tube_images_t, dim=0)) #added tensor: torch.Size([16, 224, 224, 3])
            video_images_raw.append(tube_raw_clip_images) #added PIL image
            # print('video_images[-1]: ', video_images[-1].size())
            #Box extracted from tube
            tube_box = None
            if self.cfg.BOX_STRATEGY == MIDDLE_BOX:
                m = int(len(tube_boxes)/2) #middle box from tube
                ##setting id to box
                tube_box = tube_boxes_t[m]
                id_tensor = torch.tensor([0]).unsqueeze(dim=0).float()
                # print('\n', ' id_tensor: ', id_tensor,id_tensor.size())
                # print(' c_box: ', c_box, c_box.size(), ' index: ', m)
                if tube_box.size(0)==0:
                    print(' Here error: ', path, index, '\n',
                            tube_box, '\n', 
                            sampled_tube, '\n', 
                            frames_indices, '\n', 
                            tube_boxes_t, len(tube_boxes_t), '\n', 
                            tube_boxes, len(tube_boxes), '\n',
                            t_combination)
                    exit()
                f_box = torch.cat([id_tensor , tube_box], dim=1).float()
            elif self.cfg.BOX_STRATEGY == UNION_BOX:
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
            elif self.cfg.BOX_STRATEGY == ALL_BOX:
                f_box = [torch.cat([torch.tensor([i]).unsqueeze(dim=0), torch.from_numpy(t)], dim=1).float() for i, t in enumerate(tube_boxes)]
                f_box = torch.stack(f_box, dim=0)
            final_tube_boxes.append(f_box)
        
        #load keyframes
        key_frames = []
        if self.config['input_2'] is not None:
            for k in range(len(video_images_raw)):
                if self.cfg.KEYFRAME_STRATEGY == DYNAMIC_IMAGE_KEYFRAME:
                    # key_frame, _ = self.load_input_2_di(sampled_frames_indices[k], path, frames_names_list)
                    key_frame = self.dynamic_image_fn(video_images[k])
                    if self.config['input_2']['spatial_transform']:
                        key_frame = self.config['input_2']['spatial_transform'](key_frame)
                else:
                    if self.cfg.KEYFRAME_STRATEGY == RGB_MIDDLE_KEYFRAME:
                        m = int(video_images[k].size(0)/2) #using frames loaded from 3d branch
                        key_frame = video_images[k][m] #tensor 
                        key_frame = key_frame.numpy()
                        if self.config['input_2']['spatial_transform']:
                            key_frame = self.config['input_2']['spatial_transform'](key_frame)
                    else:
                        #TODO
                        print('Not implemented yet...')
                        exit()
                key_frames.append(key_frame)
        
        #padding
        if len(video_images)<self.cfg.NUM_TUBES:
            for i in range(self.cfg.NUM_TUBES-len(video_images)):
                video_images.append(video_images[len(video_images)-1])
                p_box = tube_boxes[len(tube_boxes)-1]
                tube_boxes.append(p_box)
                if self.config['input_2'] is not None:
                    key_frames.append(key_frames[-1])

        final_tube_boxes = torch.stack(final_tube_boxes, dim=0).squeeze()
        
        if len(final_tube_boxes.shape)==1:
            final_tube_boxes = torch.unsqueeze(final_tube_boxes, dim=0)
            # print('boxes unsqueeze: ', boxes)
        
        video_images = torch.stack(video_images, dim=0).permute(0,4,1,2,3)#.permute(0,2,1,3,4)
        if self.config['input_2'] is not None:
            key_frames = torch.stack(key_frames, dim=0)
            if torch.isnan(key_frames).any().item():
                print('Detected Nan at: ', path)
            if torch.isinf(key_frames).any().item():
                print('Detected Inf at: ', path)
            # print('video_images: ', video_images.size())
            # print('key_frames: ', key_frames.size())
            # print('final_tube_boxes: ', final_tube_boxes,  final_tube_boxes.size())
            # print('label: ', label)
            return final_tube_boxes, video_images, label, num_tubes, path, key_frames
        else:
            return final_tube_boxes, video_images, label, num_tubes, path


