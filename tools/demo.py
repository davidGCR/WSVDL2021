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

class VideoDemo:
    def __init__(self, path, tub_cfg, mot_cgf, tub_file=None, ped_file=None):
        self.path = Path(path)
        self.check_file(self.path)
        # parts = self.path.parts
        # self.dataset = parts[-5]
        # self.clase = parts[-2]
        # self.set = parts[-3]
        # if tub_file:
        #     self.tub_file = Path(tub_file)/self.dataset/self.set/self.clase/self.path.stem+'.json'
        
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
        # plot_create_save_dirs(config)
        self.tub_cfg['plot_config']['plot_tubes'] = True
        self.tub_cfg['plot_config']['debug_mode'] = False
    
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
    
    def detect(self, mode):
        tubes = None
        if self.tub_file:
            print('Loading tubes from: ', self.tub_file)
            tubes = JSON_2_tube(str(self.tub_file))
            indices, names = self.temporal_step()
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

    vd = VideoDemo(
        path=r"C:\Users\David\Desktop\DATASETS\RWF-2000\frames\train\Fight\0_DzLlklZa0_5",
        # tub_file=r"C:\Users\David\Desktop\DATASETS\ActionTubesV2\RWF-2000\train\Fight\0_DzLlklZa0_5.json",
        tub_cfg=TUBE_BUILD_CONFIG,
        mot_cgf=MOTION_SEGMENTATION_CONFIG,
        ped_file=r"C:\Users\David\Desktop\DATASETS\PersonDetections\RWF-2000\train\Fight\0_DzLlklZa0_5.json"
    )

    print(vd.frames)
    # print(vd.num_frames)
    # print(vd.temporal_step()[1])
    # print(vd.root)  
    # print(vd.tub_file)

    vd.detect(0)
    # print(vd.detect(0))

if __name__=='__main__':
    demo()