import glob

common_path = "/home/cassio/CrowdedDataset/DetectionFiles/"
path_to_save = common_path + "Yolov5/Yolov5l/Yolov5l-clean-scoxyXY/"
new_thresh = 0.00
# Grab original detections
det_file_names = []
for filename in glob.glob(common_path + "Yolov5/Yolov5l/Results/*.txt"):
    det_file_names.append(filename)
# For each original detection file
for det_file in det_file_names:
    # Open the original detection file and get all lines without \n
    with open(det_file) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # Get file name only, without the path
    file_name = det_file.split("/")[-1]
    i = 0
    with open(path_to_save + file_name, "w") as out:
        for l in lines:
            l = l.split(" ")
            # Get score
            score = float(l[1])
            # Correct negatives
            xmin = 0 if int(l[2]) < 0 else int(l[2])
            ymin = 0 if int(l[3]) < 0 else int(l[3])
            xmax = 0 if int(l[4]) < 0 else int(l[4])
            ymax = 0 if int(l[5]) < 0 else int(l[5])
            if xmin >= xmax or ymin >= ymax:
                print("{} -> {} {} {} {} {}".format(file_name, score, xmin, ymin, xmax, ymax))
                continue
            # If score greater or equal than new thresh keep it
            if score >= new_thresh:
                if i == 0:
                    out.write('{:s} {:s} {:d} {:d} {:d} {:d}'.format(l[0], l[1], xmin, ymin, xmax, ymax))
                else:
                    out.write('\n{:s} {:s} {:d} {:d} {:d} {:d}'.format(l[0], l[1], xmin, ymin, xmax, ymax))
                i += 1
