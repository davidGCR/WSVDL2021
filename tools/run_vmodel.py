from os import error
import add_libs
from configs.defaults import get_cfg_defaults
from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam

from utils.utils import get_torch_device, load_checkpoint, save_checkpoint
from utils.global_var import *
from utils.create_log_name import log_name

from datasets.make_dataset_handler import load_make_dataset, load_make_dataset_UCFCrime2Local
from datasets.dataloaders import data_with_tubes, data_with_tubes_localization

from lib.optimization import train, val
from lib.optimization_mil import train_regressor, val_regressor
from lib.accuracy import calculate_accuracy_2

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def main(h_path):
    # Setup cfg.
    cfg = get_cfg_defaults()
    
    # cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_DYNIMG.yaml")
    cfg.merge_from_file(WORK_DIR / "configs/TWOSTREAM_16RGB_MIL.yaml")
    cfg.ENVIRONMENT.DATASETS_ROOT = h_path
    print(cfg)

    # test_dataset(cfg)
    # exit()


    device = get_torch_device()
    if cfg.MODEL._HEAD == BINARY:
        make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        train=True,
                                        category=2,
                                        shuffle=False)
        make_dataset_val = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        train=False,
                                        category=2,
                                        shuffle=False)                           
        train_loader, val_loader = data_with_tubes(cfg, make_dataset_train, make_dataset_val)
    
    elif cfg.MODEL._HEAD == REGRESSION:
        make_dataset_train = load_make_dataset(cfg.DATA,
                                        env_datasets_root=cfg.ENVIRONMENT.DATASETS_ROOT,
                                        train=True,
                                        category=2,
                                        shuffle=False)
        make_dataset_val = load_make_dataset_UCFCrime2Local(Path(cfg.ENVIRONMENT.DATASETS_ROOT))
        train_loader, TWO_STREAM_INPUT_val = data_with_tubes_localization(cfg, make_dataset_train)
        # from debug_loc_dataset import debug_ucfcrime2localclips_dataset
        # debug_ucfcrime2localclips_dataset(make_dataset_val, TWO_STREAM_INPUT_val)
    
    # exit()

    model = TwoStreamVD_Binary_CFam(cfg.MODEL).to(device)
    params = model.parameters()
    exp_config_log = log_name(cfg)

    #log
    h_p = HOME_DRIVE if cfg.ENVIRONMENT.DATASETS_ROOT==HOME_COLAB else cfg.ENVIRONMENT.DATASETS_ROOT
    tsb_path_folder = os.path.join(h_p, PATH_TENSORBOARD, exp_config_log)
    chk_path_folder = os.path.join(h_p, PATH_CHECKPOINT, exp_config_log)
    for p in [tsb_path_folder, chk_path_folder]:
        if not os.path.exists(p):
            os.makedirs(p)
    # print('tensorboard dir:', tsb_path)                                                
    writer = SummaryWriter(tsb_path_folder)

    if  cfg.SOLVER.OPTIMIZER.NAME == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            params, 
            lr=cfg.SOLVER.LR, 
            eps=1e-8)
    elif cfg.SOLVER.OPTIMIZER.NAME == 'SGD':
        optimizer = torch.optim.SGD(params=params,
                                    lr=cfg.SOLVER.LR,
                                    momentum=0.9,
                                    weight_decay=1e-3)
    elif cfg.SOLVER.OPTIMIZER.NAME == 'Adam':
        optimizer = torch.optim.Adam(
            params, 
            lr=cfg.SOLVER.LR, 
            eps=1e-3, 
            amsgrad=True)            
    
    if cfg.SOLVER.CRITERION == 'CEL':
        criterion = nn.CrossEntropyLoss().to(device)
    elif cfg.SOLVER.CRITERION == 'BCE':
        criterion = nn.BCELoss().to(device)
    
    start_epoch = 0
    ##Restore training
    if cfg.MODEL.RESTORE_TRAIN:
        print('Restoring training from: ', cfg.MODEL.CHECKPOINT_PATH)
        model, optimizer, epochs, last_epoch, last_loss = load_checkpoint(model, device, optimizer, cfg.MODEL.CHECKPOINT_PATH)
        start_epoch = last_epoch+1
        # config.num_epoch = epochs
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        verbose=True,
        factor=cfg.SOLVER.OPTIMIZER.FACTOR,
        min_lr=cfg.SOLVER.OPTIMIZER.MIN_LR)
    
    for epoch in range(start_epoch, cfg.SOLVER.EPOCHS):
        if cfg.MODEL._HEAD == BINARY:
            train_loss, train_acc, train_time = train(
                train_loader, 
                epoch, 
                model, 
                criterion, 
                optimizer, 
                device, 
                cfg.TUBE_DATASET.NUM_TUBES, 
                calculate_accuracy_2)
            writer.add_scalar('training loss', train_loss, epoch)
            writer.add_scalar('training accuracy', train_acc, epoch)
            
            val_loss, val_acc = val(
                val_loader,
                epoch, 
                model, 
                criterion,
                device,
                cfg.TUBE_DATASET.NUM_TUBES,
                calculate_accuracy_2)
            scheduler.step(val_loss)
            writer.add_scalar('validation loss', val_loss, epoch)
            writer.add_scalar('validation accuracy', val_acc, epoch)
        elif cfg.MODEL._HEAD == REGRESSION:
            train_loss, train_acc = train_regressor(
                train_loader, 
                epoch, 
                model, 
                criterion, 
                optimizer, 
                device, 
                cfg.TUBE_DATASET.NUM_TUBES, 
                None,
                False)
            
            ap05, ap02 = val_regressor(cfg.TUBE_DATASET,
                                       make_dataset_val, 
                                       TWO_STREAM_INPUT_val, 
                                       model, 
                                       device, 
                                       epoch,
                                       Path(cfg.ENVIRONMENT.DATASETS_ROOT)/"UCFCrime2Local/UCFCrime2LocalClips",
                                       Path(cfg.ENVIRONMENT.DATASETS_ROOT)/"ActionTubesV2/UCFCrime2LocalClips")
            
            # scheduler.step(train_loss)
            # writer.add_scalar('training loss', train_loss, epoch)
            writer.add_scalar('AP-0.5', ap05, epoch)
            writer.add_scalar('AP-0.2', ap02, epoch)


        if (epoch+1)%cfg.SOLVER.SAVE_EVERY == 0:
            save_checkpoint(model, cfg.SOLVER.EPOCHS, epoch, optimizer,train_loss, os.path.join(chk_path_folder,"save_at_epoch-"+str(epoch)+".chk"))

if __name__=='__main__':
    h_path = HOME_OSX
    torch.autograd.set_detect_anomaly(True)
    main(h_path)
