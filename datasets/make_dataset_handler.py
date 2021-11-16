from utils.global_var import *
from datasets.make_dataset import MakeRWF2000, MakeHockeyDataset, MakeRLVDDataset
from datasets.make_UCFCrime import MakeUCFCrime

def load_make_dataset(cfg,
                      train=True,
                      category=2, 
                      shuffle=False):
    """[summary]

    Args:
        cfg (yaml): cfg.DATA
        train (bool, optional): [description]. Defaults to True.
        category (int, optional): [description]. Defaults to 2.
        shuffle (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    home_path =     cfg.ROOT
    at_path =       cfg.ACTION_TUBES_FOLDER
    dataset_name =  cfg.DATASET
    cv_split =      cfg.CV_SPLIT
    load_gt =       cfg.LOAD_GROUND_TRUTH
    if dataset_name == RWF_DATASET:
        make_dataset = MakeRWF2000(
            root=os.path.join(home_path, 'RWF-2000/frames'),
            train=train,
            category=category, 
            # path_annotations=os.path.join(home_path, at_path, 'final/rwf'),
            path_annotations=os.path.join(home_path, at_path, 'RWF-2000'),
            shuffle=shuffle)

    elif dataset_name == HOCKEY_DATASET:
        make_dataset = MakeHockeyDataset(
            root=os.path.join(home_path, 'HockeyFightsDATASET/frames'), 
            train=train,
            cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/hockey_jpg{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
            path_annotations=os.path.join(home_path, at_path, 'final/hockey'),
            )
    elif dataset_name == RLVSD_DATASET:
        make_dataset = MakeRLVDDataset(
            root=os.path.join(home_path, 'RealLifeViolenceDataset/frames'), 
            train=train,
            cv_split_annotation_path=os.path.join(home_path, 'VioNetDB-splits/RealLifeViolenceDataset{}.json'.format(cv_split)), #'/content/DATASETS/VioNetDB-splits/hockey_jpg{}.json'
            path_annotations=os.path.join(home_path, at_path, 'RealLifeViolenceDataset'),
            )
    # elif dataset_name == UCFCrime_DATASET:
    #     ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl') if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
    #     make_dataset = MakeUCFCrime(
    #         root=os.path.join(home_path, 'UCFCrime/frames'), 
    #         sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
    #         sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
    #         action_tubes_path=os.path.join(home_path,'ActionTubes/UCFCrime_reduced', ann_file[1]),
    #         train=train,
    #         ground_truth_tubes=False)
    elif dataset_name == UCFCrimeReduced_DATASET:
        ann_file  = ('Train_annotation.pkl', 'Train_normal_annotation.pkl') if train else ('Test_annotation.pkl', 'Test_normal_annotation.pkl')
        make_dataset = MakeUCFCrime(
            root=os.path.join(home_path, 'UCFCrime_Reduced', 'frames'), 
            sp_abnormal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[0]), 
            sp_normal_annotations_file=os.path.join(home_path,'VioNetDB-splits/UCFCrime', ann_file[1]), 
            action_tubes_path=os.path.join(home_path, at_path, 'UCFCrime_Reduced'),
            train=train,
            ground_truth_tubes=load_gt)

    return make_dataset