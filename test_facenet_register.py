import glob
import os
import ssl
import uuid
import numpy as np
import torch
import yaml
from PIL import Image
from PIL import ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import detection_profile
import detection_eyeblink_mouthopen
import detection_emotion
import detection_liveness

ssl._create_default_https_context = ssl._create_unverified_context

profile_detector = detection_profile.detect_face_orientation()
emotion_detector = detection_emotion.predict_emotions()
blink_detector = detection_eyeblink_mouthopen.eyeblink_mouthopen_detector()


class DeployConfig:
    def __init__(self, conf_file):
        if not os.path.exists(conf_file):
            raise Exception('Config file path [%s] invalid!' % conf_file)

        with open(conf_file) as fp:
            configs = yaml.load(fp, Loader=yaml.FullLoader)
            deploy_conf = configs["FACE"]
            self.face_db = deploy_conf["FACE_DB"]
            self.threshold = deploy_conf["THRESHOLD"]


class FaceRecognition:
    @staticmethod
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

    def __init__(self, conf_file):
        self.config = DeployConfig(conf_file)
        self.font = ImageFont.truetype("fonts/simfang.ttf", 20, encoding="utf-8")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.register_mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                                    factor=0.709, post_process=True, device=self.device)
        self.recognition_mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # 人脸库的人脸特征
        if os.path.exists("model/db_facenet.pt"):
            self.faces_embeddings = torch.load("model/db_facenet.pt")
        else:
            self.faces_embeddings = list()

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        exist_userids = list()
        files = glob.glob(face_db_path + "/*.jpg")
        for faces_embedding in self.faces_embeddings:
            exist_userids.append(faces_embedding["userid"])
        for file in files:
            userid = file.split("/")[1].split(".")[0]
            if userid not in exist_userids:
                face = Image.open(file)
                face_aligns = []
                face_align, prob = self.register_mtcnn(face, return_prob=True)
                if face_align is not None:
                    print('Face detected with probability: {:8f}'.format(prob))
                    face_aligns.append(face_align)
                    face_aligns = torch.stack(face_aligns).to(self.device)
                    embedding = self.resnet(face_aligns).detach().cpu()
                    self.faces_embeddings.append({
                        "userid": userid,
                        "feature": embedding
                    })

    @staticmethod
    def find_feature(feature1, faces_embeddings, threshold):
        if len(faces_embeddings) <= 0:
            return -1
        features = []
        for i in range(len(faces_embeddings)):
            features.append(faces_embeddings[i]['feature'])

        probs = [(feature1 - features[i]).norm().item() for i in range(len(features))]
        if min(probs) < threshold:
            return probs.index(min(probs))
        else:
            return -1

    def register(self, image, username):
        # 加载人脸库中的人脸
        self.load_faces(self.config.face_db)
        face_align, prob = self.register_mtcnn(image, return_prob=True)
        face_aligns = []
        if face_align is None:
            return None
        face_aligns.append(face_align)
        face_aligns = torch.stack(face_aligns).to(self.device)
        # 判断人脸是否存在
        embedding = self.resnet(face_aligns).detach().cpu()
        find_index = self.find_feature(embedding, self.faces_embeddings, self.config.threshold)
        if find_index >= 0:
            return self.faces_embeddings[find_index]['userid']

        uuidstring = ''.join(str(uuid.uuid4()).split('-'))
        userid = username + '_' + uuidstring
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        Image.fromarray(np.asarray(image)).save(os.path.join(self.config.face_db, f'{userid}.jpg'))
        self.faces_embeddings.append({
            "userid": userid,
            "feature": embedding
        })
        return userid

    def recognition(self, image):
        faces = self.recognition_mtcnn(image)
        boxes, probs = self.recognition_mtcnn.detect(image)
        if faces is not None and boxes is not None:
            results = list()
            for i in range(len(boxes)):
                bbox = [int(boxes[i].tolist()[0]), int(boxes[i].tolist()[1]), int(boxes[i].tolist()[2]), int(boxes[i].tolist()[3])]
                result = detection_liveness.detect_liveness(image, bbox)
                embedding = self.resnet(faces[i].unsqueeze(0).to(self.device))
                result["userid"] = "unknown"
                find_index = self.find_feature(embedding, self.faces_embeddings, self.config.threshold)
                if find_index >= 0:
                    result["userid"] = self.faces_embeddings[find_index]['userid']
                results.append(result)
            return results
        else:
            return None


if __name__ == "__main__":
    filepaths = glob.glob('datasets/train_images/*/1.jpg')
    face_recognition = FaceRecognition("config_facenet.yaml")

    for filepath in filepaths:
        img = Image.open(filepath)
        userid = face_recognition.register(img, filepath.split('/')[2])
        print(userid)
    torch.save(face_recognition.faces_embeddings, 'model/db_facenet.pt')
