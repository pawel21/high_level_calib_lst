import glob
from tools import create_dl1b_tailcut

data_list_dl1 = sorted(glob.glob("/media/pawel1/ADATA HD330/20201122/DL1/v0.6.3_v05/dl1*"))

for path in data_list_dl1:
    new_path = 'data/'+'dl1b.' + path.split('/')[-1][10:23] + "_8_4.h5"
    print(new_path)
    create_dl1b_tailcut(path, new_path, "lstchain_standard_config.json")
