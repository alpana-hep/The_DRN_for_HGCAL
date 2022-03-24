import pickle
import awkward as ak

#feat_list = ["Hit_X.pickle", "Hit_Y.pickle", "Hit_Z.pickle", "SsLocation.pickle", "beamEn.pickle", "logratioflip_target.pickle", "ratio_target.pickle", "ratioflip_target.pickle", "rawE.pickle", "recHitEn.pickle", "trueE.pickle", "trueE_target.pickle"]

feat_list = ["Hit_X.pickle", "Hit_Y.pickle", "Hit_Z.pickle", "logratioflip_target.pickle", "ratio_target.pickle", "ratioflip_target.pickle", "rawE.pickle", "recHitEn.pickle", "trueE.pickle", "trueE_target.pickle","SsLocation.pickle"]


for feat in feat_list:
        print("combining "+feat+".....")
        var = feat
	
        # with open("test_0to5M_fix_wt/%s"%var, 'rb') as f:
        #     trueE0_5 = pickle.load(f)
        # print("5M read")

        with open("./../../pickle_0to1M/%s"%var, 'rb') as f:
            trueE0 = pickle.load(f)
        print(6)

        with open("./../../pickle_1to2M/%s"%var, 'rb') as f:
            trueE1 = pickle.load(f)
        print(7)


        with open("./../../pickle_2to3M/%s"%var, 'rb') as f:
            trueE2 = pickle.load(f)
        print(8)


        with open("./../../pickle_3to4M/%s"%var, 'rb') as f:
            trueE3 = pickle.load(f)
        print(9)


        with open("./../../pickle_4to5M/%s"%var, 'rb') as f:
            trueE4 = pickle.load(f)
        print(10)
        print()



        comb_var = ak.concatenate([trueE0, trueE1, trueE2, trueE3, trueE4])

        with open("/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_cartfeat_2ipfeat_z_en/%s"%var, 'wb') as f:
                pickle.dump(comb_var, f, protocol=4)

import torch
print("combining cartfeat...")

# var0_5 = torch.load("test_0to5M_fix_wt/cartfeat.pickle")
# print("0-5 done")

var1 = torch.load("./../../pickle_0to1M/cartfeat.pickle")
print(6)
var2 = torch.load("./../../pickle_1to2M/cartfeat.pickle")
print(7)
var3 = torch.load("./../../pickle_2to3M/cartfeat.pickle")
print(8)
var4 = torch.load("./../../pickle_3to4M/cartfeat.pickle")
print(9)
var5 = torch.load("./../../pickle_4to5M/cartfeat.pickle")
print(10)
comb_var = var1 + var2 + var3 + var4 + var5

with open("/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_cartfeat_2ipfeat_z_en/cartfeat.pickle", 'wb') as f:
            torch.save(comb_var, f,pickle_protocol = 4)

#v_comb = torch.load("test_0to10M_fix_raw_ahcalTrim/cartfeat.pickle")
print("len of comb_cartfeat=",len(comb_var))


print("creating train/val index")

import numpy as np

#proportion of dataset to use as training set
split = 0.8

####################################################
# Main logic                                       #
####################################################

folder = "/home/rusack/shared/pickles/HGCAL_TestBeam/pkl_files/pkl_cartfeat_2ipfeat_z_en/"

#a bit silly, but load in trueE to figure out data length
with open("%s/rawE.pickle"%folder, 'rb') as f:
    trueE = pickle.load(f)

length = len(trueE)
print("len(rawE)=", length)
print("unique rawE=", np.unique(np.asarray(trueE)).size)

#create train/test split
train_idx = np.random.choice(length, int(split * length + 0.5), replace=False)
mask = np.ones(length, dtype=bool)
mask[train_idx] = False
valid_idx = mask.nonzero()[0]

with open("%s/all_valididx.pickle"%folder, 'wb') as f:
    pickle.dump(valid_idx, f)

with open("%s/all_trainidx.pickle"%folder, 'wb') as f:
    pickle.dump(train_idx, f)

print("len trainidx=",train_idx.size)
print("unique trainidx=", np.unique(train_idx).size)

print("Done....")
