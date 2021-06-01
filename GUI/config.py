import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='/content/drive/MyDrive/Retinal_Vessel_Segmentation/Experiments',
                        help='trained model will be saved at here') #Not use
    parser.add_argument('--save', default='UNet_vessel_seg',
                        help='save name of experiment in args.outf directory') #Not use
    parser.add_argument('--best_model_path', default='G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/Code/Retinal_Vessel_Segmentation/GUI/best_model/best_model.pth',
                        help='directory of best model path')
    # model parameters
    parser.add_argument('--in_channels', default=1,type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2,type=int, 
                        help='output channels of model')
    parser.add_argument('--batch_size', default=64,
                       type=int, help='batch size')
    # inference
    parser.add_argument('--test_patch_height', default=64)
    parser.add_argument('--test_patch_width', default=64)
    parser.add_argument('--stride_height', default=16)
    parser.add_argument('--stride_width', default=16)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')
    args = parser.parse_args()

    return args
