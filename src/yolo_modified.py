import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class PersonDetector(Node):

    def __init__(self):
        super().__init__('person_detector')
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()
        self.net = cv2.dnn.readNet('/home/sunny/Desktop/yolov3-tiny.weights', '/home/sunny/Desktop/yolov3-tiny.cfg')
        self.classes = []
        with open('/home/sunny/Desktop/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = self.net.getUnconnectedOutLayersNames()

    def image_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, channels = cv_image.shape

        # Object detection
        blob = cv2.dnn.blobFromImage(cv_image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Post-processing
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indices) > 0:
            indices = indices.flatten()
            boxes = [boxes[i] for i in indices]
            confidences = [confidences[i] for i in indices]
            class_ids = [class_ids[i] for i in indices]

        # Draw bounding boxes
        for i in range(len(boxes)):
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_image, 'person', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
"""
        # Show image
        cv2.imshow('Person Detection', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    person_detector = PersonDetector()
    rclpy.spin(person_detector)
    person_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""