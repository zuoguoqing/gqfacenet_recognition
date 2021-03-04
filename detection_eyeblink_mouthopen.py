import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

cfg_ear_threshold = 0.23
cfg_mar_threshold = 0.3
cfg_face_landmarks = "model/shape_predictor_68_face_landmarks.dat"

class eyeblink_mouthopen_detector:
    def __init__(self):
        self.predictor = dlib.shape_predictor(cfg_face_landmarks)

    def mouth_open(self, gray, rect):
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        mar = self.mouth_aspect_ratio(shape[mStart:mEnd])
        if mar > cfg_mar_threshold:
            return True
        else:
            return False

    def eye_blink(self, gray, rect):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = self.predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < cfg_ear_threshold:
            return True
        else:
            return False

    @staticmethod
    def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        result = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return result

    @staticmethod
    def mouth_aspect_ratio(mouth):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[1], mouth[7])
        B = dist.euclidean(mouth[2], mouth[6])
        C = dist.euclidean(mouth[3], mouth[5])

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        D = dist.euclidean(mouth[0], mouth[4])
        # compute the eye aspect ratio
        result = (A + B + C) / (2.0 * D)
        # return the eye aspect ratio
        return result
