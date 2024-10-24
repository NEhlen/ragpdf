from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load a model
model = YOLO("models/yolo11n.pt")

# Train the model
train_results = model.train(
    data="data/data.yaml",  # path to dataset YAML
    epochs=300,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    patience=200,
    batch=-1,
)

# # Perform object detection on an image
results = model("data/test/montage_7740995-0.png")
img = results[0].plot()
plt.imshow(img)
plt.show()
