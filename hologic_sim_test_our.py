import os
import argparse
import torch
import time
import h5py
import numpy as np
from utils import *
from network_our import ResNet
from DataLoader_hologic_sim import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="DBT_tensor_test")

parser.add_argument("--data_path", type=str, default="data/",
                    help='path to training data')
parser.add_argument("--logdir", type=str, default="model/hologic_sim_model_our_3/net_latest.pth",
                    help='path to model and log files')
parser.add_argument("--save_path", type=str,
                    default="results/hologic_sim_model_our_3/", help='path to save results')
parser.add_argument("--use_GPU", type=bool,
                    default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="8", help='GPU id')

opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    
    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = ResNet()

    if opt.use_GPU:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join(opt.logdir)))

    model.eval()
    
    # load data info
    print('Loading data info ...\n')
    time_test = 0

    dataset_val = DataSet_DBT(opt.data_path,phase='val',tensor_if=True)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    print("# of validating samples: %d" % int(len(dataset_val)))

    f_output = h5py.File(opt.save_path+'test.h5', "w")

    i=0
   
    for num, item in enumerate(loader_val, 0):
        input = Variable(torch.Tensor(item['input']))

        if opt.use_GPU:
            input = input.cuda()
        with torch.no_grad():
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()

            output = model(input)

            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()

            dur_time = end_time - start_time
            time_test += dur_time
            print('output_'+str(i), ' cost time: ', dur_time)  

        if opt.use_GPU:
            save_out = de_normalize_NF(np.float32(output[0].data.cpu().numpy().squeeze()))  # back to cpu
            # save_out = de_normalize_NF(np.float32(output[1].data.cpu().numpy().squeeze()))  # back to cpu
        else:
            save_out = de_normalize_NF(np.float32(output[0].data.numpy().squeeze()))
            # save_out = de_normalize_NF(np.float32(output[1].data.numpy().squeeze()))

        f_output.create_dataset('output_'+str(i), data=save_out, dtype= 'float32')
        
        i=i+1

    f_output.close()

    print('Avg. time:', time_test/(i))  


if __name__ == "__main__":
    main()
