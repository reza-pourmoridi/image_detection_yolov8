from ultralytics import YOLO

# Load a model
model = YOLO(r'C:\Users\reza\Desktop\yolo\yolo8\runs\detect\train\weights\best.pt')  # pretrained YOLOv8n model

image_paths = [
    r'C:\Users\reza\Desktop\yolo\yolo8\datasets\test\images\image_10044.jpg',
    # r'C:\Users\reza\Desktop\yolo\yolo8\datasets\train\images\image_2.jpg',
    # r'C:\Users\reza\Desktop\yolo\yolo8\datasets\train\images\image_3.jpg'
]
# Run batched inference on a single image (corrected)
results = model(image_paths)  # return a list of Results objects (single image)

# Process results list

for i, result in enumerate(results):
    print("Image:", image_paths[i])  # Access image name using its index
    print(result)
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.show()  # Display to screen
