import os
import re
import torch.utils.data as data
from operator import itemgetter
import numpy as np
import cv2
import torch
# from torchvision import transforms
# from TubeletGeneration.metrics import extract_tubes_from_video
# from TubeletGeneration.tube_utils import JSON_2_videoDetections
# from VioNet.customdatasets.make_dataset import MakeUCFCrime2LocalClips
# from VioNet.dataset import video_loader
# from VioNet.utils import natural_sort


from utils.dataset_utils import imread
from utils.tube_utils import JSON_2_videoDetections
from utils.utils import natural_sort

class MakeUCFCrime2LocalClips():
    def __init__(self, root, path_annotations, path_person_detections, abnormal):
        self.root = root
        self.path_annotations = path_annotations
        self.path_person_detections = path_person_detections
        self.classes = ['normal', 'anomaly'] #Robbery,Stealing
        self.subclasses = ['Arrest', 'Assault'] #Robbery,Stealing
        self.abnormal = abnormal
    
    def __get_list__(self, path):
        paths = os.listdir(path)
        paths = [os.path.join(path,pt) for pt in paths if os.path.isdir(os.path.join(path,pt))]
        return paths
    
    def __annotation__(self, folder_path):
        v_name = os.path.split(folder_path)[1]
        annotation = [ann_file for ann_file in os.listdir(self.path_annotations) if ann_file.split('.')[0] in v_name.split('(')]
        annotation = annotation[0]
        return os.path.join(self.path_annotations, annotation)

    
    def ground_truth_boxes(self, video_folder, ann_path):
        frames = os.listdir(video_folder)
        frames_numbers = [int(re.findall(r'\d+', f)[0]) for f in frames]
        frames_numbers.sort()
        # print(frames_numbers)

        annotations = []
        with open(ann_path) as fid:
            lines = fid.readlines()
            ss = 1 if lines[0].split()[5] == '0' else 0
            for line in lines:
                # v_name = line.split()[0]
                # print(line.split())
                ann = line.split()
                frame_number = int(ann[5]) + ss
                valid = ann[6]
                if valid == '0' and frame_number in frames_numbers:
                    annotations.append(
                        {
                            "frame": frame_number,
                            "xmin": ann[1],
                            "ymin": ann[2],
                            "xmax": ann[3],
                            "ymax": ann[4]
                        }
                    )
        return annotations
    
    def plot(self, folder_imgs, annotations_dict, live_paths=[]):
        imgs = os.listdir(folder_imgs)
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [ atoi(c) for c in re.split(r'(\d+)', text) ]
        
        imgs.sort(key=natural_keys)
        # print(type(folder_imgs),type(f_paths[0]))
        f_paths = [os.path.join(folder_imgs, ff) for ff in imgs]
        
        for img_path in f_paths:
            print(img_path)
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            f_num = os.path.split(img_path)[1]
            f_num = int(re.findall(r'\d+', f_num)[0])
            ann = [ann for ann in annotations_dict if ann['frame']==f_num][0]
            x1 = ann["xmin"]
            y1 = ann["ymin"]
            x2 = ann["xmax"]
            y2 = ann["ymax"]
            cv2.rectangle(image,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0,238,238),
                            1)
            if len(live_paths)>0:
                frame = img_path.split('/')[-1]
                
                for l in range(len(live_paths)):
                    
                    foundAt = True if frame in live_paths[l]['frames_name'] else False
                    if foundAt:
                        idx = live_paths[l]['frames_name'].index(frame)
                        bbox = live_paths[l]['boxes'][idx]
                        x1 = bbox[0]
                        y1 = bbox[1]
                        x2 = bbox[2]
                        y2 = bbox[3]
                        cv2.rectangle(image,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (255,0,0),
                                    1)
            cv2.namedWindow('FRAME'+str(f_num),cv2.WINDOW_NORMAL)
            cv2.resizeWindow('FRAME'+str(f_num), (600,600))
            image = cv2.resize(image, (600,600))
            cv2.imshow('FRAME'+str(f_num), image)
            key = cv2.waitKey(250)#pauses for 3 seconds before fetching next image
            if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()

    def __call__(self):
        root_anomaly = os.path.join(self.root, self.classes[1])
        root_normal = os.path.join(self.root, self.classes[0])
        
        if self.abnormal:
            abnormal_paths = self.__get_list__(root_anomaly)
            paths = abnormal_paths
            annotations_anomaly = [self.__annotation__(pt) for pt in abnormal_paths]
            annotations = annotations_anomaly
            labels = [1]*len(abnormal_paths)
            annotations_p_detections = []
            num_frames = []
            for ap in abnormal_paths:
                assert os.path.isdir(ap), 'Folder does not exist!!!'
                n = len(os.listdir(ap))
                num_frames.append(n)
                sp = ap.split('/')
                p_path = os.path.join(self.path_person_detections, sp[-2], sp[-1]+'.json')
                assert os.path.isfile(p_path), 'P_annotation does not exist!!!'
                annotations_p_detections.append(p_path)

        else:
            normal_paths = self.__get_list__(root_normal)
            normal_paths = [path for path in normal_paths if "Normal" in path]
            paths = normal_paths
            annotations_normal = [None]*len(normal_paths)
            annotations = annotations_normal
            labels = [0]*len(normal_paths)
            annotations_p_detections = [None]*len(normal_paths)
            num_frames = []
            for ap in normal_paths:
                assert os.path.isdir(ap), 'Folder does not exist!!!'
                n = len(os.listdir(ap))
                num_frames.append(n)
        # paths = abnormal_paths + normal_paths
        # annotations = annotations_anomaly + annotations_normal
        # labels = [1]*len(abnormal_paths) + [0]*len(normal_paths)
        
        return paths, labels, annotations, annotations_p_detections, num_frames

class UCFCrime2LocalDataset(data.Dataset):
    """
    Load tubelets from one video
    Use to extract features tube-by-tube from just a video
    """

    def __init__(
        self, 
        root,
        path_annotations,
        abnormal,
        persons_detections_path,
        transform=None,
        clip_len=25,
        clip_temporal_stride=1):
        # self.dataset_root = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
        # self.split = 'anomaly',
        # self.video = 'Arrest036(2917-3426)',
        # self.p_d_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
        # self.gt_ann_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos'

        # self.dataset_root = dataset_root
        # self.split = split
        # self.video = video
        # self.p_d_path = p_d_path
        # self.gt_ann_path = gt_ann_path
        # self.transform = transform
        # self.person_detections = JSON_2_videoDetections(p_d_path)
        # self.tubes = extract_tubes_from_video(
        #     self.dataset_root,
        # )
        self.clip_temporal_stride = clip_temporal_stride
        self.clip_len = clip_len
        self.root = root
        self.path_annotations = path_annotations
        self.abnormal = abnormal
        self.make_dataset = MakeUCFCrime2LocalClips(root, path_annotations, abnormal)
        self.paths, self.labels, self.annotations = self.make_dataset()

        self.persons_detections_path = persons_detections_path

    def __len__(self):
        return len(self.paths)
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]

        return indices_segments

    def generate_tube_proposals(self, path, frames):
        tmp = path.split('/')
        split = tmp[-2]
        video = tmp[-1]
        p_d_path = os.path.join(self.persons_detections_path, split, video)
        person_detections = JSON_2_videoDetections(p_d_path)
        tubes = extract_tubes_from_video(
            self.root,
            person_detections,
            frames,
            # {'wait': 200}
            )
        return tubes

    def __getitem__(self, index):
        path = self.paths[index]
        ann = self.annotations[index]
        sp_annotations_gt = self.make_dataset.ground_truth_boxes(path, ann)

        video_clips = self.get_video_clips(path)
        return video_clips, path, ann, sp_annotations_gt

class UCFCrime2LocalVideoDataset(data.Dataset):
    def __init__(
        self, 
        path,
        sp_annotation,
        # p_detections,
        transform=None,
        clip_len=25,
        clip_temporal_stride=1,
        tubes=None,
        transformations=None):
        self.path = path
        self.sp_annotation = sp_annotation
        # self.p_detections = p_detections
        self.transform = transform
        self.clip_len = clip_len
        self.clip_temporal_stride = clip_temporal_stride

        self.clips = self.get_video_clips(self.path)
        self.video_name = path.split('/')[-1]
        self.clase = path.split('/')[-2]
        self.tubes = tubes
        self.transformations = transformations
    
    def __len__(self):
        return len(self.clips)
    
    def split_list(self, lst, n):  
        for i in range(0, len(lst), n): 
            yield lst[i:i + n] 
    
    def get_video_clips(self, video_folder):
        _frames = os.listdir(video_folder)
        _frames = [f for f in _frames if '.jpg' in f]
        num_frames = len(_frames)

        # indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        # indices_segments = [indices[x:x + self.clip_len] for x in range(0, len(indices), self.clip_len)]
        # real_clip_len = self.clip_len*self.clip_temporal_stride
        indices = [x for x in range(0, num_frames, self.clip_temporal_stride)]
        indices_segments = list(self.split_list(indices, self.clip_len)) 

        return indices_segments
    
    def load_frames(self, indices):
        image_names = os.listdir(self.path)
        image_names = natural_sort(image_names)
        image_names = list(itemgetter(*indices)(image_names))
        image_paths = [os.path.join(self.path,img_name) for img_name in image_names]
        images = []
        for ip in image_paths:
            img = self.transform(imread(ip)) if self.transform else imread(ip)
            images.append(img)
        # print('len(images): ', len(images), type(images[0]))
        images = torch.stack(images, dim=0)
        return image_names, images
    
    def load_frames_from_numbers(self, indices):
        image_names = ['frame{:03}.jpg'.format(n) for n in indices]
        image_paths = [os.path.join(self.path,img_name) for img_name in image_names]
        images = []
        for ip in image_paths:
            img = imread(ip)
            images.append(img)
        # print('len(images): ', len(images), type(images[0]))
        # images = torch.stack(images, dim=0)
        return image_names, images
    
    def load_sp_annotations(self, frames, ann_path):
        frames_numbers = [int(re.findall(r'\d+', f)[0]) for f in frames]
        frames_numbers.sort()
        annotations = []
        with open(ann_path) as fid:
            lines = fid.readlines()
            ss = 1 if lines[0].split()[5] == '0' else 0
            for line in lines:
                # v_name = line.split()[0]
                # print(line.split())
                ann = line.split()
                frame_number = int(ann[5]) + ss
                valid = ann[6]
                if valid == '0' and frame_number in frames_numbers:
                    annotations.append(
                        {
                            "frame": frame_number,
                            "xmin": ann[1],
                            "ymin": ann[2],
                            "xmax": ann[3],
                            "ymax": ann[4]
                        }
                    )
        
        return annotations
    
    def __centered_frames__(self, tube_frames_idxs, tube_len, max_video_len, min_frame):
        if len(tube_frames_idxs) == tube_len: 
            return tube_frames_idxs
        # else:
        #     tube_frames_idxs = np.linspace(min_frame, max_video_len, 16).astype(int)
        #     tube_frames_idxs = tube_frames_idxs.tolist()
        #     return tube_frames_idxs
        if len(tube_frames_idxs) > tube_len:
            n = len(tube_frames_idxs)
            m = int(n/2)
            arr = np.array(tube_frames_idxs)
            centered_array = arr[m-int(tube_len/2) : m+int(tube_len/2)]
            return centered_array.tolist()
        if len(tube_frames_idxs) < tube_len: #padding

            # print('padding...')
            center_idx = int(len(tube_frames_idxs)/2)
            
            start = tube_frames_idxs[center_idx]-int(tube_len/2)
            end = tube_frames_idxs[center_idx]+int(tube_len/2)
            out = list(range(start,end))
            # print('center_idx: {}, val:{}, start:{}, end:{}={}'.format(center_idx, tube_frames_idxs[center_idx], start, end, out))
            if out[0]<min_frame:
                most_neg = abs(out[0])
                out = [i+most_neg for i in out]
            elif tube_frames_idxs[center_idx]+int(tube_len/2) > max_video_len:
                start = tube_frames_idxs[center_idx]-(tube_len-(max_video_len-tube_frames_idxs[center_idx]))+1
                end = max_video_len+1
                out = list(range(start,end))
            tube_frames_idxs = out
            return tube_frames_idxs
    
    def __central_bbox__(self, tube, id):
        width, height = 224, 224
        if len(tube)>2:
            central_box = tube[int(len(tube)/2)]
        else:
            central_box = tube[0]
        central_box = central_box[0:4]
        central_box = np.array([max(central_box[0], 0), max(central_box[1], 0), min(central_box[2], width - 1), min(central_box[3], height - 1)])
        central_box = np.insert(central_box[0:4], 0, id).reshape(1,-1)
        central_box = torch.from_numpy(central_box).float()
        return central_box
    
    def get_tube_data(self, tube, max_num_frames, min_frame, box_id):
        sampled_frames_from_tubes = []
        bboxes_from_tubes = []
        
        tube_real_frames = [int(re.search(r'\d+', fname).group()) for fname in tube['frames_name']]
        # print('==tube_real_frames: ', tube_real_frames, len(tube_real_frames))
        centered_frames = self.__centered_frames__(
            tube_real_frames,
            16,
            max_num_frames,
            min_frame
        )
        # print('==centered_frames: ', centered_frames, len(centered_frames))
        image_names, images = self.load_frames_from_numbers(centered_frames)
        
        bbox = self.__central_bbox__(tube['boxes'], box_id)
        keyframe = images[int(len(images)/2)]
        # keyframe = torch.unsqueeze(torch.tensor(keyframe), dim=0)
        
        
        if self.transformations['input_1']['spatial_transform'] is not None:
            images = self.transformations['input_1']['spatial_transform'](images)
        if self.transformations['input_2']['spatial_transform'] is not None:
            keyframe = self.transformations['input_2']['spatial_transform'](keyframe)

        images = torch.stack(images)
        images = torch.unsqueeze(images, dim=0).permute(0,2,1,3,4)
        keyframe = torch.unsqueeze(keyframe, dim=0)
        # print('images: ', type(images))

        return images, bbox, keyframe

    def __getitem__(self, index):
        clip = self.clips[index]
        # image_names, images = self.load_frames(clip)
        image_names = os.listdir(self.path)
        image_names = natural_sort(image_names)
        image_names = list(itemgetter(*clip)(image_names))
        gt = self.load_sp_annotations(image_names, self.sp_annotation)
        
        return clip, image_names, gt, len(clip)

if __name__=='__main__':
    # val_make_dataset = MakeUCFCrime2LocalClips(
    #         root='/media/david/datos/Violence DATA/UCFCrime2LocalClips/UCFCrime2LocalClips',
    #         # root_normal='/Volumes/TOSHIBA EXT/DATASET/AnomalyCRIMEALL/UCFCrime2Local/frames',
    #         path_annotations='/media/david/datos/Violence DATA/UCFCrime2LocalClips/Txt annotations-longVideos',
    #         path_person_detections='/media/david/datos/Violence DATA/PersonDetections/ucfcrime2local',
    #         abnormal=True)
    # paths, labels, annotations, annotations_p_detections, num_frames = val_make_dataset()

    # dataset = UCFCrime2LocalDataset(
    #     root='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips',
    #     path_annotations='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos',
    #     abnormal=True,
    #     persons_detections_path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/ucfcrime2local',
    #     transform=None,
    #     clip_len=25,
    #     clip_temporal_stride=5)
    
    
    # video_clips, path, ann, sp_annotations_gt = dataset[45]
    # print(path)
    # print(ann)
    # print(video_clips)

    # video_dataset = UCFCrime2LocalVideoDataset(
    #     path='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/UCFCrime2Local/UCFCrime2LocalClips/anomaly/Stealing091(245-468)',
    #     sp_annotation='/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CrimeViolence2LocalDATASET/Txt annotations-longVideos/Stealing091.txt',
    #     p_detections='',
    #     transform=transforms.ToTensor(),
    #     clip_len=25,
    #     clip_temporal_stride=5
    # )

    # for clip, frames, gt in video_dataset:
    #     print('--',clip, len(clip), frames.size())
    #     for g in gt:
    #         print(g)
    print()