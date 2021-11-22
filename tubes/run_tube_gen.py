from tubes.incremental_linking import IncrementalLinking
from tubes.motion_segmentation import MotionSegmentation
import time

def extract_tubes_from_video(frames_indices, frames_names, motion_seg_config, tube_build_config, gt=None):
    """Extract violent action tubes from a video

    Args:
        frames_indices (list): Indices of the frames in the folder.
        frames_names (list): Names of the files/images. They must be in order.
        motion_seg_config (dict): Configuration settings for motion segmentation.
        tube_build_config (dict): Configuration settings for tube building.
        gt (bool, optional): Flac to plot ground truth. Defaults to None.

    Returns:
        tuple: (live_paths, time) List of action tubes and execution time.
    """
    segmentator = MotionSegmentation(motion_seg_config)
    tube_builder = IncrementalLinking(tube_build_config)
    start = time.time()
    live_paths = tube_builder(frames_indices, frames_names, segmentator, gt)
    end = time.time()
    exec_time = end - start
    return  live_paths, exec_time