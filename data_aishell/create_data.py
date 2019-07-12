import os
import tqdm

def covert_data(from_file, to_file, from_dir="", from_sp="\t"):
    nones = 0
    with open(from_file) as in_f:
        with open(to_file) as out_f:
            for line in tqdm.tqdm(in_f):
                if line.strip()=="": continue
                fp, label = line.strip().split(from_sp, 1)
                fp = from_dir + fp
                if os.path.exists(fp):
                    out_f.write(fp + "\t" + label + "\n")
                else:
                    nones += 1
    print(nones)

covert_data("/home/d")