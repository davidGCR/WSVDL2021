from yacs.config import CfgNode as CN


_C = CN()

# _C.SYSTEM = CN()
# # Number of GPUS to use in the experiment
# _C.SYSTEM.NUM_GPUS = 8
# # Number of workers for doing things
# _C.SYSTEM.NUM_WORKERS = 4

# _C.TRAIN = CN()
# # A very important hyperparameter
# _C.TRAIN.HYPERPARAMETER_1 = 0.1
# # The all important scales for the stuff
# _C.TRAIN.SCALES = (2, 4, 8, 16)


_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.DATASETS_ROOT = ""

#NAme of the class to use in log files
_C.MODEL = CN()
_C.MODEL.NAME = ""
_C.MODEL.WITH_ROIPOOL = False
_C.MODEL._HEAD = ""
_C.MODEL.RESTORE_TRAIN = False
_C.MODEL.CHECKPOINT_PATH = ""

_C.MODEL._3D_BRANCH = CN()
_C.MODEL._3D_BRANCH.NAME="i3d" #i3d, 3dresnet
_C.MODEL._3D_BRANCH.FINAL_ENDPOINT="Mixed_4e" #'Mixed_5b', #Mixed_4e, so far, only for i3d
_C.MODEL._3D_BRANCH.PRETRAINED_MODEL = ""
_C.MODEL._3D_BRANCH.FREEZE_3D=False

_C.MODEL._ROI_LAYER = CN()
_C.MODEL._ROI_LAYER.OUTPUT = 0
_C.MODEL._ROI_LAYER.WITH_TEMPORAL_POOL = False
_C.MODEL._ROI_LAYER.SPATIAL_SCALE = 0
_C.MODEL._ROI_LAYER.WITH_SPATIAL_POOL = False
_C.MODEL._ROI_LAYER.TYPE = 'RoIAlign'


_C.MODEL._2D_BRANCH = CN()
_C.MODEL._2D_BRANCH.NAME="resnet50"
_C.MODEL._2D_BRANCH.FINAL_ENDPOINT="layer3"
_C.MODEL._2D_BRANCH.NUM_TRAINABLE_LAYERS=3

_C.MODEL._CFAM_BLOCK = CN()
_C.MODEL._CFAM_BLOCK.IN_CHANNELS=0 #528+1024 #528+1024,#528+1024, #832+2048
_C.MODEL._CFAM_BLOCK.OUT_CHANNELS = 0

_C.MODEL._FC = CN()
_C.MODEL._FC.INPUT_DIM = 0


# from pathlib import Path
#Dataset folders and splits
_C.DATA = CN()
_C.DATA.ROOT = "/media/david/datos/Violence DATA"
_C.DATA.DATASET = ""
_C.DATA.CV_SPLIT = -1
_C.DATA.SPLITS_FOLDER = "VioNetDB-splits"
_C.DATA.ACTION_TUBES_FOLDER = "ActionTubesV2"
_C.DATA.LOAD_GROUND_TRUTH = False

# _C.RWF_DATASET = CN()
# _C.RWF_DATASET.NAME = "RWF-2000"
# _C.RWF_DATASET.ROOT = "RWF-2000/frames"

# _C.HOCKEY_DATASET = CN()
# _C.HOCKEY_DATASET.NAME = "HockeyFightsDATASET"
# _C.HOCKEY_DATASET.ROOT = "HockeyFightsDATASET/frames"

# _C.RLVSD_DATASET = CN()
# _C.RLVSD_DATASET.NAME = "RealLifeViolenceDataset"
# _C.RLVSD_DATASET.ROOT = "RealLifeViolenceDataset/frames"

_C.UCFCRIME_DATASET = CN()
_C.UCFCRIME_DATASET.NAME = "UCFCrime_Reduced"
_C.UCFCRIME_DATASET.ROOT = "UCFCrime_Reduced/frames"
_C.UCFCRIME_DATASET.TRAIN_ANNOT_ABNORMAL = "Train_annotation.pkl"
_C.UCFCRIME_DATASET.TRAIN_ANNOT_NORMAL = "Train_normal_annotation.pkl"
_C.UCFCRIME_DATASET.TEST_ANNOT_ABNORMAL = "Test_annotation.pkl"
_C.UCFCRIME_DATASET.TEST_ANNOT_NORMAL = "Test_normal_annotation.pkl"

#Tube dataset config
_C.TUBE_DATASET = CN()
_C.TUBE_DATASET.NUM_FRAMES = 16
_C.TUBE_DATASET.NUM_TUBES = 4
_C.TUBE_DATASET.RANDOM = True
# _C.TUBE_DATASET.DATASET = ""
_C.TUBE_DATASET.FRAMES_STRATEGY = -1
_C.TUBE_DATASET.BOX_STRATEGY = -1
_C.TUBE_DATASET.KEYFRAME_STRATEGY = -1
_C.TUBE_DATASET.SHAPE = [224,224]
_C.TUBE_DATASET.MAKE_FN = None
# _C.TUBE_DATASET.TRAIN = False
_C.TUBE_DATASET.DATALOADERS_DICT = False
_C.TUBE_DATASET.BOX_AS_TENSOR = False

#Dataloaderconfig
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH = 4
_C.DATALOADER.VAL_BATCH = 4
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.DROP_LAST = True

#log paths
_C.LOGS = CN()
_C.LOGS.TENSORBOARD_FOLDER = ""
_C.LOGS.CHECKPOINTS_FOLDER = ""
_C.LOGS._FOLDER = ""

#solver config
_C.SOLVER = CN()
_C.SOLVER.CRITERION = ""
_C.SOLVER.LR = 0.
_C.SOLVER.EPOCHS = 1
_C.SOLVER.SAVE_EVERY = 1

_C.SOLVER.OPTIMIZER = CN()
_C.SOLVER.OPTIMIZER.NAME = ""
_C.SOLVER.OPTIMIZER.FACTOR = 0.1
_C.SOLVER.OPTIMIZER.MIN_LR = 1e-7

_C.TUBE_GENERATOR = CN()

_C.TUBE_GENERATOR.INC_LINKING = CN()
_C.TUBE_GENERATOR.INC_LINKING.TRAIN_MODE = True
_C.TUBE_GENERATOR.INC_LINKING.IMG_SIZE = [224,224]
_C.TUBE_GENERATOR.INC_LINKING.CLOSE_PERSONS_REP = 10
_C.TUBE_GENERATOR.INC_LINKING.TMP_WINDOW = 5
_C.TUBE_GENERATOR.INC_LINKING.IOU_CLOSE_PERSONS = 0.3
_C.TUBE_GENERATOR.INC_LINKING.JUMPGAP = 5
_C.TUBE_GENERATOR.INC_LINKING.MIN_WINDOW_LEN = 3

_C.TUBE_GENERATOR.INC_LINKING.PLOT = CN()
_C.TUBE_GENERATOR.INC_LINKING.PLOT.DEBUG = False
_C.TUBE_GENERATOR.INC_LINKING.PLOT.TUBES = False
_C.TUBE_GENERATOR.INC_LINKING.PLOT.WAIT_TUBES = 100
_C.TUBE_GENERATOR.INC_LINKING.PLOT.WAIT_2 = 100
_C.TUBE_GENERATOR.INC_LINKING.PLOT.SAVE_RESULTS = False
_C.TUBE_GENERATOR.INC_LINKING.PLOT.SAVE_FOLDER_DEBUG = ""
_C.TUBE_GENERATOR.INC_LINKING.PLOT.SAVE_FOLDER_FINAL = ""

_C.TUBE_GENERATOR.INC_LINKING.DATA = CN()
_C.TUBE_GENERATOR.INC_LINKING.DATA.ROOT = ""
_C.TUBE_GENERATOR.INC_LINKING.DATA.PER_DETECTIONS = ""


_C.TUBE_GENERATOR.MOTION_SEG = CN()
_C.TUBE_GENERATOR.MOTION_SEG.BINARY_THRES = 150
_C.TUBE_GENERATOR.MOTION_SEG.MIN_CONECTED_COMP_AREA = 49


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()