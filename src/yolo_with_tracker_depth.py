import rclpy
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import cv2
from centroidtracker import CentroidTracker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

class PersonDetector(Node):

    def __init__(self):
        super().__init__('person_detector')
        self.rgb_subscription = self.create_subscription(
            Image,
            '/color/preview/image',
            self.image_callback,
            10)
        
        self.depth_subscription = self.create_subscription(
            Image,
            '/stereo/depth',
            self.depth_image_callback,
            10)
        
        self.rgb_subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()
        self.net = cv2.dnn.readNet('/home/sunny/Desktop/yolov3-tiny.weights', '/home/sunny/Desktop/yolov3-tiny.cfg')
        self.classes = []
        with open('/home/sunny/Desktop/coco.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        self.tracker = CentroidTracker()
        self.ct = CentroidTracker()
        self.centroid_publisher = self.create_publisher(Point, '/centroid_data', 10)
        #self.depth_publisher = self.create_publisher(Point, '/depth_data', 10)
        self.target = None
        self.depth_meters = 0
        
        
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
                if confidence > 0.7 and class_id == 0:
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
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.1)
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
            #else:
                #print('no  centroid')   
        # Show image
        #cv2.imshow('Person Detection', cv_image)
        #cv2.waitKey(1)

        # Pass bounding boxes to centroid tracker
        rects = []
        for box in boxes:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            rects.append((x, y, x+w, y+h))

        #print(rects)
    
        objects = self.ct.update(rects)

        

        #print(objects)

        if len(objects) > 0:
            first_object_ID = list(objects.keys())[0]
            self.target = list(objects.values())[0]
            #print("Object ID of first object:", first_object_ID)
            #print("Centroid of first object:", self.target)
            #print("depth of first object:", self.depth_meters)

            
            self.target_y = self.target[1]
            
            centroid_msg = Point()
            centroid_msg.x =  float(self.target[0])
            centroid_msg.y =  float(self.target[1])
            centroid_msg.z =  float(self.depth_meters)
           
            self.centroid_publisher.publish(centroid_msg)
        else:
            self.target = None

        
       
        # Draw tracked objects
        for (objectID, centroid) in objects.items():
            #self.last_image_time = time.time()
            text = "ID {}".format(objectID)
            cv2.putText(cv_image, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(cv_image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
        
        # Show tracked image
        cv2.imshow("Person Tracking", cv_image)
        cv2.waitKey(1)
        
    
    def depth_image_callback(self, msg):

        if self.target is not None:

            # Convert ROS Image message to OpenCV image
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Get the desired pixel values from the depth image
            x = self.target[0]  # example x coordinate
            y = self.target[1]  # example y coordinate
            depth_value = depth_image[y, x]  # assuming 8-bit depth map image

            # Convert the depth value to meters
            self.depth_meters = depth_value / 1000.0  # assuming depth map values are in millimeters
            print(f"Depth at ({x}, {y}): {self.depth_meters} m")
        
            #depth_msg = Point()
            #depth_msg.z =  float(depth_meters)
         
           
            #self.centroid_publisher.publish(depth_msg.z)

            
        else:
            
            print('no target in sight, cannot initialise tracking')
            
            

def main(args=None):
    rclpy.init(args=args)
    person_detector = PersonDetector()
    rclpy.spin(person_detector)
    person_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
        main()