import glob
import os

if __name__ == "__main__":
    CONFIG_PATH = "/pscratch/sd/d/diptarko/Thesis/fyear/main/thesis/GaugeConfigurationGeneration/configuration"
    path_list = glob.glob(os.path.join(CONFIG_PATH,"*.info"))
    max_ = 0
    for path in path_list:
        index = int(path.split("/")[-1].strip(".info").strip("conf"))
        max_ = max(max_,index)
    next_index = max_
    
    print(next_index)