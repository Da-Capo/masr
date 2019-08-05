from pathlib import Path

ai_shell_root = "/mnt/h/Downloads/AISHELL/data_aishell/"

with open(ai_shell_root+"transcript/aishell_transcript_v0.8.txt") as f:
    label_dict = {}
    for i,line in enumerate(f):
        key,value = line.strip().split(" ",1)
        label_dict[key] = value.replace(" ","")

train_f  = open("train-sort.manifest","w")
dev_f    = open("dev.manifest","w")
for f_path in Path(ai_shell_root).rglob('*.wav'):
    x = str(f_path)
    try:
        y = label_dict[f_path.name.split(".")[0]]
        if "train" in x:
            train_f.write(x+","+y+"\n")
        if "test" in x:
            dev_f.write(x+","+y+"\n")
    except:
        print(x)
train_f.close()
dev_f.close()
