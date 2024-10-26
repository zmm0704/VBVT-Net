from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
from utils import *

train_list = ["001"]

test_list = ["003"]

roi_csv_path = "hologic_sim_roi.csv"


class DataSet_DBT(Dataset):

    def __init__(self, root_dir, phase="train", tensor_if=True):
        print(" load dataset ...  ", phase)
        self.phase = phase
        self.tensor_if = tensor_if
        self.trainlist = []
        self.vallist = []
        # self.f_train = h5py.File(root_dir+'hologic_sim_train.h5', "r")
        self.f_val = h5py.File(root_dir + "hologic_sim_val.h5", "r")
        with open(roi_csv_path, "r") as f:
            f.readline()
            all_lines = f.readlines()
            for line in all_lines:
                values = line.split(",")
                if values[0] in train_list:
                    self.trainlist.append(line)
                elif values[0] in test_list:
                    self.vallist.append(line)

        if self.phase == "train":
            self.len = len(self.trainlist)
        elif self.phase == "val":
            self.len = len(self.vallist)
        else:
            print("Please give the right phase!!")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # if self.phase == 'train':
        #     f_data = self.f_train
        # elif self.phase == 'val':
        # f_data = self.f_val

        f_data = self.f_val

        if self.tensor_if:
            return {
                "input": normalize_NF(np.squeeze(f_data["input_" + str(idx)], axis=0)),
                "label": normalize_NF(np.squeeze(f_data["label_" + str(idx)], axis=0)),
            }
        else:
            return {
                "input": normalize_NF(np.squeeze(f_data["image_" + str(idx)], axis=0)),
                "label": normalize_NF(np.squeeze(f_data["label_" + str(idx)], axis=0)),
            }
