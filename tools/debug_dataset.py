
from transformations.data_aug.data_aug import *
from transformations.vizualize_batch import *
# from model_transformations import i3d_video_transf, resnet_transf

from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.make_dataset import *
from datasets.make_UCFCrime import *
from datasets.tube_dataset import *

def test_dataset(cfg):
    ann_file  = (cfg.UCFCRIME_DATASET.TRAIN_ANNOT_ABNORMAL, cfg.UCFCRIME_DATASET.TRAIN_ANNOT_NORMAL)# if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
    home_path = cfg.DATA.ROOT
    make_dataset = MakeUCFCrime(
            root=os.path.join(home_path, cfg.UCFCRIME_DATASET.ROOT), 
            sp_abnormal_annotations_file=os.path.join(home_path, cfg.DATA.SPLITS_FOLDER,'UCFCrime', ann_file[0]), 
            sp_normal_annotations_file=os.path.join(home_path, cfg.DATA.SPLITS_FOLDER, 'UCFCrime', ann_file[1]), 
            action_tubes_path=os.path.join(home_path, cfg.DATA.ACTION_TUBES_FOLDER, cfg.UCFCRIME_DATASET.NAME),
            train=True,
            ground_truth_tubes=True)
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)  
    inputs_config = {
        'input_1': {
            'type': 'rgb',
            # 'spatial_transform': i3d_video_transf()['train'],
            'spatial_transform': Compose(
                [
                    ClipRandomHorizontalFlip(), 
                    # ClipRandomScale(scale=0.2, diff=True), 
                    ClipRandomRotate(angle=5),
                    # ClipRandomTranslate(translate=0.1, diff=True),
                    NumpyToTensor()
                ],
                probs=[1, 1]
                ),
            'temporal_transform': None
        },
        # 'input_2': {
        #     'type': 'rgb',
        #     'spatial_transform': resnet_transf()['val'],
        #     'temporal_transform': None
        # }
        'input_2': {
            'type': 'dynamic-image',
            'spatial_transform': transforms.Compose([
                transforms.Resize((224,224)),
                # transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                norm
            ]),
            'temporal_transform': None
        }
    }
    # train_dataset = TubeDataset(frames_per_tube=16, 
    #                         make_function=make_dataset,
    #                         max_num_tubes=1,
    #                         train=True,
    #                         dataset='UCFCrime_Reduced',
    #                         random=True,
    #                         tube_box=UNION_BOX,
    #                         config=TWO_STREAM_INPUT_train,
    #                         key_frame=DYNAMIC_IMAGE_KEYFRAME)

    # cfg.TUBE_DATASET.MAKE_FN = make_dataset
    cfg.TUBE_DATASET.DATASET = cfg.UCFCRIME_DATASET.NAME
    # cfg.TUBE_DATASET.DATALOADERS_DICT = TWO_STREAM_INPUT_train
    cfg.TUBE_DATASET.FRAMES_STRATEGY = MIDDLE_FRAMES
    cfg.TUBE_DATASET.BOX_STRATEGY = MIDDLE_BOX
    cfg.TUBE_DATASET.KEYFRAME_STRATEGY = DYNAMIC_IMAGE_KEYFRAME
    train_dataset = TubeDataset(
                                cfg.TUBE_DATASET,
                                make_dataset,
                                inputs_config
                                )
    
    # for i in range(len(train_dataset)):
    #     data = train_dataset[i]
    #     bboxes, video_images, label, num_tubes, path, key_frames = data
    #     if os.path.split(path)[1]=='Assault027_x264':
    #         print(i)
    #         break
    # random.seed(34)
    for i in range(1):
        bboxes, video_images, label, num_tubes, path, key_frames = train_dataset[400]
        print('\tpath: ', path)
        print('\tvideo_images: ', type(video_images), video_images.size())
        print('\tbboxes: ', bboxes.size())
        print('\tkey_frames: ', type(key_frames), key_frames.size())

        frames_numpy = video_images.permute(0,2,3,4,1)
        frames_numpy = frames_numpy.cpu().numpy().astype('uint8')
        bboxes_numpy = bboxes.cpu().numpy()[:,1:5] #remove id and to shape (n,4)\
        key_frames_numpy = key_frames.permute(0,2,3,1)
        key_frames_numpy = key_frames_numpy.cpu().numpy()
        
        print('\nframes_numpy: ', frames_numpy.shape)
        print('bboxes_numpy: ', bboxes_numpy)
        print('key_frames_numpy: ', key_frames_numpy.shape)

        for j in range(frames_numpy.shape[0]): #iterate over batch
            
            bboxes_numpy = [bboxes_numpy] * 16
            plot_clip(frames_numpy[j], bboxes_numpy, (4,4))
            plot_keyframe(key_frames_numpy[j], bboxes_numpy[0])