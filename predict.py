import os
import csv
import pandas

image_folder_path = r"C:\Users\reza\Desktop\yolo\yolo8\datasets\test\images"
output_csv_path = r"C:\Users\reza\Desktop\yolo\yolo8\submission.csv"

images_pathes_list = []
images_names_list = []
for image_name in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image_name)
    images_pathes_list.append(image_path)
    cleaned_image_name = image_name.split('.')[0]
    images_names_list.append(cleaned_image_name)

# images_pathes_list = [
#     r'C:\Users\reza\Desktop\yolo\yolo8\datasets\train\images\image_1.jpg',
#     r'C:\Users\reza\Desktop\yolo\yolo8\datasets\train\images\image_2.jpg',
#     r'C:\Users\reza\Desktop\yolo\yolo8\datasets\train\images\image_3.jpg'
# ]
#
# images_names_list = [
#     r'image_1.jpg',
#     r'image_2.jpg',
#     r'image_3.jpg'
# ]

if images_pathes_list:
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImageId', 'PredictionString'])  # Write header row


        from ultralytics import YOLO
        model = YOLO(r'C:\Users\reza\Desktop\yolo\yolo8\runs\detect\train\weights\best.pt')  # pretrained YOLOv8n model
        results = model(images_pathes_list)  # return a list of Results objects (single image)

        for i, result in enumerate(results):
            name = images_names_list[i]
            boxes = result.boxes  # Access bounding box data
            prediction_string = ""
            for box in boxes:
                label = int(box.cls[0])  # Get the class label (integer)
                confidence = box.conf[0]  # Get the confidence score
                x_center, y_center, width, height = box.xywhn[0]  # Extract coordinates and dimensions

                x_center_str = "{:.2f}".format(x_center)
                y_center_str = "{:.2f}".format(y_center)
                width_str = "{:.2f}".format(width)
                height_str = "{:.2f}".format(height)
                prediction_string += f"{label} {confidence:.1f} {x_center_str} {y_center_str} {width_str} {height_str} "

            writer.writerow([name, prediction_string.strip()])  # Remove trailing whitespace
            print(f"ImageID: {name}")
            print(f"PredictionString: {prediction_string.strip()}")