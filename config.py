from configparser import ConfigParser as CP

cp = CP()
cp.read("config.ini")
SKIP = cp['Default'].getint('skip_frames')
SHOW = cp['Default'].getboolean('show')
detector_name = cp['Default']['detector_name']

if detector_name == "FaceboxesTensorflow":
    from detectors.tensorflow_detectors import FaceboxesTensorflow
    DETECTOR = FaceboxesTensorflow(cp['FaceboxesTensorflow']['weights_path'],
                                   cp['FaceboxesTensorflow'].getfloat('score_threshold'))
elif detector_name == "OpenCVHaar":
    from detectors.opencv_detectors import OpenCVHaarFaceDetector
    DETECTOR = OpenCVHaarFaceDetector()
