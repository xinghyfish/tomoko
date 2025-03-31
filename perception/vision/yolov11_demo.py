from PIL import Image
from ultralytics import YOLO

from perception.vision.image_utils import show_box_on_image
from perception.vision.vision_model import VisionModel

yolo = YOLO("../checkpoints/yolov11/yolo11x.pt", task="detect")
results = yolo(source="../assets/teacups.jpg", save=False)

def test_yolo():
    for result in results:
        save_dir = result.save_dir
        print(save_dir)
        boxes = result.boxes
        for box in boxes:
            # print(box)
            # Get the confidence score and category index
            conf = box.conf.item()  # Probability of the detection
            cls = box.cls.item()  # Class index of the detection

            # Get class name (optional, if you have class names available)
            class_name = yolo.names[int(cls)]  # Convert class index to class name
            x_min, y_min, x_max, y_max = list(map(int, box.xyxy.cpu().numpy()[0]))

            # Print or store the result
            print(f"Category: {class_name}, Confidence: {conf}, box: {[x_min, y_min, x_max, y_max]}")


def test_box():
    image = Image.open("../assets/teacups.jpg").convert("RGB")
    class_name = "cup"
    vm = VisionModel('small', 'x')
    detect_info = vm.detect(image, class_name)
    show_box_on_image(image, class_name, detect_info)


if __name__ == '__main__':
    test_box()
    