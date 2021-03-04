import cv2
import dlib
import numpy as np
import detection_eyeblink_mouthopen
import detection_emotion
import detection_profile
from paz.pipelines import DetectMiniXceptionFER

def bboverlab(boxi, boxj):
    x1 = boxi[0]
    y1 = boxi[1]
    w1 = boxi[2] - boxi[0]
    h1 = boxi[3] - boxi[1]
    x2 = boxj[0]
    y2 = boxj[1]
    w2 = boxj[2] - boxj[0]
    h2 = boxj[3] - boxj[1]
    if x1 > x2 + w2:
        return False
    elif y1 > y2 + h2:
        return False
    elif x1 + w1 < x2:
        return False
    elif y1 + h1 < y2:
        return False
    return True

profile_detector = detection_profile.detect_face_orientation()
eyeblink_mouthopen_detector = detection_eyeblink_mouthopen.eyeblink_mouthopen_detector()
emotion_detector = DetectMiniXceptionFER([0.1, 0.1])
emotion2_detector = detection_emotion.predict_emotions()

# image:PL.Image
# bbox:[1,2,3,4]
def detect_liveness(image, bbox):
    emotions = emotion_detector(np.array(image))
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    result = dict()
    result["bbox"] = bbox
    result["emotion"] = "neutral"
    for emotion in emotions['boxes2D']:
        if bboverlab(result["bbox"], np.array(emotion.coordinates)):
            result["emotion"] = str(emotion.class_name)
    result["emotion2"] = emotion2_detector.get_emotion(np.array(image), [result["bbox"]])
    box_profile, profile = profile_detector.face_orientation(gray)
    result["box_profile"] = box_profile
    result["profile"] = profile
    rect = dlib.rectangle(result["bbox"][0], result["bbox"][1], result["bbox"][2], result["bbox"][3])
    iseyeblink = eyeblink_mouthopen_detector.eye_blink(gray, rect)
    ismouthopen = eyeblink_mouthopen_detector.mouth_open(gray, rect)
    result["iseyeblink"] = iseyeblink
    result["ismouthopen"] = ismouthopen
    return result
