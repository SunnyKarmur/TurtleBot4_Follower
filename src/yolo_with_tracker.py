import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from centroidtracker import CentroidTracker


class PersonDetector(Node):

    def __init__(self):
        super().__init__('person_detector')
        self.subscription = self.create_subscription(
            Image,
            '/color/preview/image',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()
        self.net = cv2.dnn.readNet('/home/sunny/Desktop/yolov3-tiny.weights', '/home/sunny/Desktop/yolov3-tiny.cfg')
        self.classes = []
        with open('/home/sunny/Desktop/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.tracker = CentroidTracker()
        self.ct = CentroidTracker()

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

        # Track persons using centroid tracker
        objects = self.tracker.update(boxes)
        for i, box in enumerate(boxes):
            centroid = objects.get(i, None)
            if centroid is not None:
                # Draw the bounding box and label for the person
                x, y, w, h = box
                color = (0, 255, 0)
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(cv_image, f"Person {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw the centroid
                cv2.circle(cv_image, (int(centroid[0]), int(centroid[1])), 4, color, -1)

        # Show image
        cv2.imshow('Person Detection', cv_image)
        cv2.waitKey(1)

        # Pass bounding boxes to centroid tracker
        rects = []
        for box in boxes:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            rects.append((x, y, x+w, y+h))
    
        objects = self.ct.update(rects)
    
        # Draw tracked objects
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(cv_image, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(cv_image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            print(centroid)
    
        # Show tracked image
        cv2.imshow("Person Tracking", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    person_detector = PersonDetector()
    rclpy.spin(person_detector)
    person_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
        main()