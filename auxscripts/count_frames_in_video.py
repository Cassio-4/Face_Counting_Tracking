import cv2

def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0
	# if the override flag is passed in, revert to the manual
	# method of counting frames
	if override:
		total = count_frames_manual(video)
	# otherwise, let's try the fast way first
	else:
		# lets try to determine the number of frames in a video
		# via video properties; this method can be very buggy
		# and might throw an error based on your OpenCV version
		# or may fail entirely based on your which video codecs
		# you have installed
		try:
			total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		except:
			total = count_frames_manual(video)
	# release the video file pointer
	video.release()
	# return the total number of frames in the video
	return total


def count_frames_manual(video):
	return 9999999

print("Choke1")
print(count_frames("/home/cassio/CrowdedDataset/choke1_bboxes.avi"))
print("Choke2")
print(count_frames("/home/cassio/CrowdedDataset/choke2_bboxes_faceOnly.avi"))
print("Street")
print(count_frames("/home/cassio/CrowdedDataset/Street_CutVideo.avi"))
print("Sidewalk")
print(count_frames("/home/cassio/CrowdedDataset/Sidewalk_CutVideo.avi"))
print("Bengal")
print(count_frames("/home/cassio/CrowdedDataset/Bengal_CutVideo.avi"))
print("Terminal1")
print(count_frames("/home/cassio/CrowdedDataset/Terminal1_CutVideo.avi"))
print("Terminal2")
print(count_frames("/home/cassio/CrowdedDataset/Terminal2_CutVideo.avi"))
print("Terminal3")
print(count_frames("/home/cassio/CrowdedDataset/Terminal3_CutVideo.avi"))
print("Terminal4")
print(count_frames("/home/cassio/CrowdedDataset/Terminal4_CutVideo.avi"))
print("Shibuya")
print(count_frames("/home/cassio/CrowdedDataset/Shibuya_CutVideo.avi"))

print("MOT17-01")
print(count_frames("/home/cassio/CrowdedDataset/VideoDataset/MOT17-01_video.avi"))
print("MOT17-04")
print(count_frames("/home/cassio/CrowdedDataset/VideoDataset/MOT17-04_video.avi"))
print("MOT17-09")
print(count_frames("/home/cassio/CrowdedDataset/VideoDataset/MOT17-09_video.avi"))
