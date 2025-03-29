import os
from typing import Dict, List, Tuple, Any

from PIL import Image
from lang_sam import LangSAM
from ultralytics import YOLO

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

SAM_MODEL_PREFIX = "sam2.1_hiera_%s"
DIR_PATH = os.path.dirname(__file__)
SAM_MODEL_PATH = DIR_PATH + "/checkpoints/sam2.1/sam2.1_hiera_%s.pt"
YOLOv11_DETECT_MODEL_PATH = DIR_PATH + "/checkpoints/yolov11/yolo11%s.pt"
YOLOv11_CLASSIFICATION_MODEL_PATH = DIR_PATH + "/checkpoints/yolov11/yolo11%s-cls.pt"
CLASSIFICATION_TASK_TYPE = "classification"
DETECTION_TASK_TYPE = "detection"


class VisionModel:
    def __init__(self, sam_type='small', yolo_type='x'):
        """
        Initialize Vision Model by SAM2.1 type and YOLOv11 type.
        :param sam_type: Type of SAM to use (tiny, small, base_plus, large)
        :param yolo_type:
        """
        self.sam_type = sam_type
        self.yolo_type = yolo_type
        full_sam_model_type = SAM_MODEL_PREFIX % self.sam_type
        sam_model_path = SAM_MODEL_PATH % self.sam_type
        yolo_cls_model_path = YOLOv11_CLASSIFICATION_MODEL_PATH % self.yolo_type
        yolo_detect_model_path = YOLOv11_DETECT_MODEL_PATH % self.yolo_type
        self.lang_sam = LangSAM(full_sam_model_type, sam_model_path)
        self.yolo_detect = YOLO(yolo_detect_model_path, task=DETECTION_TASK_TYPE)
        self.yolo_classification = YOLO(yolo_cls_model_path, task=CLASSIFICATION_TASK_TYPE)

    def segment(self, image: Image, text_prompt: str) -> dict[str, Any]:
        results = self.lang_sam.predict([image], [text_prompt])
        # results.shape == (n, h, w), where n == #objects(in text prompt)
        assert 1 == len(results)
        result = results[0]
        return result

    def detect(self, image: Image) -> List[Dict]:
        results = self.yolo_detect(source=image)
        detected_objects = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                xyxy = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = xyxy
                # Get the confidence score and category index
                conf = box.conf.item()  # Probability of the detection
                cls = box.cls.item()  # Class index of the detection

                # Get class name (optional, if you have class names available)
                class_name = self.yolo_detect.names[int(cls)]  # Convert class index to class name

                info = {
                    "category": class_name,
                    "confidence": conf,
                    "box": [x_min, y_min, x_max, y_max],
                }
                detected_objects.append(info)

            return detected_objects

    def classify(self, image: Image) -> Tuple[str, float, List[Dict]]:
        results = self.yolo_classification(source=image)
        detected_objects = []

        for result in results:
            probs = result.probs
            # Index of the most probable class
            max_prob_idx = probs.argmax()
            # Probability of the most probable class
            max_prob = probs[max_prob_idx]
            # Class name corresponding to the index
            class_name = self.yolo_classification.names[max_prob_idx]

            # Optionally, print all class probabilities
            detected_objects.append({
                "category": class_name,
                "probability": max_prob,
            })
            for idx, prob in enumerate(probs):
                detected_objects.append({
                    "category": self.yolo_classification.names[idx],
                    "probability": prob,
                })
            return class_name, max_prob, detected_objects

    def count(self, image: Image, category: str) -> int:
        """
        Count the number of detected objects.
        """
        detected_objects = self.detect(image)
        count = 0
        for result in detected_objects:
            if result["category"] == category:
                count += 1
        return count

    def contain(self, image: Image, expected_category: str) -> bool:
        """
        return if the expected category object is in the image.
        """
        _, _, detected_objects = self.classify(image)
        return any([obj["category"] == expected_category for obj in detected_objects])

    def verify_number(self, image: Image, category: str, text_prompt: str, expected_number) -> bool:
        """
        Verify if the given text prompt matches the expected number.
        :param image: image sampled from video stream
        :param category: pre-detected object category
        :param text_prompt: specific text prompt of object
        :param expected_number: expected number of the scene
        :return:
        """
        masks = self.segment(image, text_prompt)['masks']
        yolo_count = self.count(image, category)
        if yolo_count < expected_number:
            return False
        else:
            gdino_count = masks.shape[0]
            return gdino_count == expected_number


if __name__ == '__main__':
    demo_image = Image.open("./assets/4cups.jpg").convert("RGB")
    vision_model = VisionModel("small", "x")
    print(vision_model.verify_number(demo_image, "cup", "cup", 4))
