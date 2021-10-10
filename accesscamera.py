import cv2
import numpy as np

# default video device
cap = cv2.VideoCapture(0)
img_name="opencv_frame.jpg"
ret, img = cap.read()
cv2.imwrite(img_name, img)
cap.release()
cv2.destroyAllWindows()
# while True:
#     # read img to var and ret to junk
#     ret, img = cap.read()
#     cv2.imshow("image",img)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# Loading image
img = cv2.imread(img_name)
# img = cv2.resizing(img, None, fx=0.9, fy=0.78)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # object has been detected '_'
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #rectangle information
            x = int(center_x - w /2)
            y = int(center_y - h /2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+30), font, 3, color, 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows(0)

# # read img to var and ret to junk
# ret, img = cap.read()
# img_name="open_cv.jpg"
# cv2.imwrite(img_name, img)
# #show image
# cv2.imshow("image",img)
# cv2.waitKey(0) #if given any key continue execution
# cv2.destroyAllWindows()# destroy all videos

