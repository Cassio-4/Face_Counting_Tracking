from configparser import ConfigParser as CP

cp = CP()
cp.read("config.ini")
SKIP = cp['Default'].getint('skip_frames')
SHOW = cp['Default'].getboolean('show')
detector_name = cp['Default']['detector_name']
tracker_name = cp['Default']['tracker_name']

# Creating the desired detector
if detector_name == "FaceboxesTensorflow":
    from detectors.tensorflow_detectors import FaceboxesTensorflow
    DETECTOR = FaceboxesTensorflow(cp['FaceboxesTensorflow']['weights'],
                                   cp['FaceboxesTensorflow'].getfloat('score_threshold'))
elif detector_name == "OpenCVHaar":
    from detectors.opencv_detectors import OpenCVHaarFaceDetector
    DETECTOR = OpenCVHaarFaceDetector(model_path=cp["OpenCVHaarCascades"]["path"])
elif detector_name == "DlibHog":
    from detectors.dlib_detectors import DlibHogDetector
    DETECTOR = DlibHogDetector()
else:
    print("Invalid Detector")
    exit(0)

# Creating desired tracker
if tracker_name == "MEDIANFLOW":
    from trackers.opencv_trackers import OpenCVMultitrackerMedianFlow
    TRACKER = OpenCVMultitrackerMedianFlow()
elif tracker_name == "BOOSTING":
    from trackers.opencv_trackers import OpenCVMultitrackerBoosting
    TRACKER = OpenCVMultitrackerBoosting()
elif tracker_name == "MIL":
    from trackers.opencv_trackers import OpenCVMultitrackerMil
    TRACKER = OpenCVMultitrackerMil()
elif tracker_name == "KCF":
    from trackers.opencv_trackers import OpenCVMultitrackerKcf
    TRACKER = OpenCVMultitrackerKcf()
elif tracker_name == "TLD":
    from trackers.opencv_trackers import OpenCVMultitrackerTld
    TRACKER = OpenCVMultitrackerTld()
elif tracker_name == "GOTURN":
    from trackers.opencv_trackers import OpenCVMultitrackerGoturn
    TRACKER = OpenCVMultitrackerGoturn()
elif tracker_name == "MOSSE":
    from trackers.opencv_trackers import OpenCVMultitrackerMosse
    TRACKER = OpenCVMultitrackerMosse()
elif tracker_name == "CSRT":
    from trackers.opencv_trackers import OpenCVMultitrackerCsrt
    TRACKER = OpenCVMultitrackerCsrt()
elif tracker_name == "DlibCorrelation":
    from trackers.dlib_trackers import DlibCorrelationTrackers
    TRACKER = DlibCorrelationTrackers()
else:
    print("Invalid tracker")
    exit(0)

if SKIP <= 0:
    print("Skip frames has to be >= 1 and int")
    exit(1)
if SKIP > 1:
    TEST_NAME = "{}-{}-{}-".format(detector_name, tracker_name, SKIP)
# If skip is 1 then what is happening is we're only running the detector
else:
    TEST_NAME = "{}-detOnly-".format(detector_name)
