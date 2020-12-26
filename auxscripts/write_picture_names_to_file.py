import os


# Navigate to folder with all videos in it
og_path = "/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/images/bar/bar001/"
# Get all paths
all_jpg = []
for r, d, f in os.walk(og_path):
    for file in f:
        if file.endswith("jpg"):
            all_jpg.append(os.path.join(r, file))

sufix = [j.split('/')[-1] for j in all_jpg]

sufix.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

with open("/home/cassio/PycharmProjects/Face_Counting_Tracking/dataset/images/bar/bar001/all_images.txt", 'w') as all:
    for s in sufix:
        all.write(s+'\n')