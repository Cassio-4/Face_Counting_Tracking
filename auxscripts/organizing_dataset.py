import os


# Navigate to folder with all frames in it
og_path = "/home/cassio/dataset/Images/P2L_S5_C3.1"
destination_path = "/home/cassio/dataset/Images/renamed-P2L_S5_C3.1/"

all_jpg = []
for r, d, f in os.walk(og_path):
    for file in f:
        if file.endswith("jpg"):
            all_jpg.append(os.path.join(r, file))

all_jpg.sort()

for i in range(len(all_jpg)):
    jpg = all_jpg[i]
    number = str(i+1)
    number = number.zfill(5)
    new_jpg = "{}.jpg".format(number)
    os.rename(jpg, destination_path+new_jpg)


