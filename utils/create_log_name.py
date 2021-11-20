def log_name(cfg):
    str_ = "{}_{}_CV({})_usingGT({})_numTubes({})_framesXtube({})_framesStrat({})_boxStrat({})_keyframeInput({})_loss({})_opt({})_lr({})_epochs({})".format(
        cfg.MODEL.NAME,
        cfg.DATA.DATASET,
        cfg.DATA.CV_SPLIT,
        cfg.DATA.LOAD_GROUND_TRUTH,
        cfg.TUBE_DATASET.NUM_TUBES,
        cfg.TUBE_DATASET.NUM_FRAMES,
        cfg.TUBE_DATASET.FRAMES_STRATEGY,
        cfg.TUBE_DATASET.BOX_STRATEGY,
        cfg.TUBE_DATASET.KEYFRAME_STRATEGY,
        cfg.SOLVER.CRITERION,
        cfg.SOLVER.OPTIMIZER.NAME,
        cfg.SOLVER.LR,
        cfg.SOLVER.EPOCHS
    )
    return str_