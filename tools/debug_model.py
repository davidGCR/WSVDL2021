import torch
from models.TwoStreamVD_Binary_CFam import TwoStreamVD_Binary_CFam

def debug_model(cfg):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TwoStreamVD_Binary_CFam(cfg).to(device)
    print('------- ViolenceDetector --------')
    
    # # model = ViolenceDetectorRegression(aggregate=True).to(device)
    batch = 1
    tubes = 3
    input_1 = torch.rand(batch*tubes,3,8,224,224).to(device)
    input_2 = torch.rand(batch*tubes,3,224,224).to(device)

    rois = torch.rand(batch*tubes, 5).to(device)
    rois[0] = torch.tensor([0,  62.5481,  49.0223, 122.0747, 203.4146]).to(device)#torch.tensor([1, 14, 16, 66, 70]).to(device)
    rois[1] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    rois[2] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[3] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[4] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[5] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[6] = torch.tensor([1, 34, 14, 85, 77]).to(device)
    # rois[7] = torch.tensor([1, 34, 14, 85, 77]).to(device)

    output = model(input_1, input_2, rois, tubes)
    # output = model(input_1, input_2, None, None)
    # output = model(input_1, rois, tubes)
    print('output: ', output.size())
