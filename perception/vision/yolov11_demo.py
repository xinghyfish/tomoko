from ultralytics import YOLO

yolo = YOLO("../checkpoints/yolov11/yolo11x.pt", task="detect")
result = yolo(source="./assets/teacups.jpg", save=True)

for result in result:
    boxes = result.boxes
    for box in boxes:
        # Get the confidence score and category index
        conf = box.conf.item()  # Probability of the detection
        cls = box.cls.item()  # Class index of the detection

        # Get class name (optional, if you have class names available)
        class_name = yolo.names[int(cls)]  # Convert class index to class name

        # Print or store the result
        print(f"Category: {class_name}, Confidence: {conf}")
