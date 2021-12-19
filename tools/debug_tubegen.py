from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset_utils import read_JSON_ann
from datasets.CCTVFights_dataset import *
from tubes.run_tube_gen import *
from configs.tube_config import TUBE_BUILD_CONFIG, MOTION_SEGMENTATION_CONFIG
from utils.tube_utils import JSON_2_videoDetections

def test_tubegen_CCTVFights_dataset():
    """Test Sequential Dataset to load long videos clip by clip. Extract action tubes from clips
    """
    root = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CCTVFights/frames/fights"
    root_person_detec = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/PersonDetections/CCTVFights"
    json_file = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/CCTVFights/groundtruth_modified.json"
    data = read_JSON_ann(json_file)
    tubes_path = "/Users/davidchoqueluqueroman/Documents/DATASETS_Local/ActionTubesV2/CCTVFights/fights"
    print(data["version"])

    paths, frame_rates, tmp_annotations, person_det_files = make_CCTVFights_dataset(root, root_person_detec, json_file, "testing")
    transform = transforms.ToTensor()

    for j, (path, frame_rate, tmp_annot, pers_detect_annot) in enumerate(zip(paths, frame_rates, tmp_annotations, person_det_files)):
        print(j, path)
        # print("tmp_annot: ", tmp_annot)
        # print("pers_detect_annot: ", pers_detect_annot)
        # print("frame_rate: ", frame_rate)
        dataset = SequentialDataset(seq_len=16, tubes_path=tubes_path, pers_detect_annot=pers_detect_annot, annotations=tmp_annot, video_path=path, frame_rate=frame_rate, transform=transform)

        loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        )
        for video_images, label, tubes in loader:
            # frames_names = [list(i) for i in zip(*frames_names)]
            print('\tprocessing clip: VIDEO_IMGS: {}, TUBES: {}'.format(video_images.size(), len(tubes)))