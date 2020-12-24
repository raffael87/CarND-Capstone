from styx_msgs.msg import TrafficLight
import tensorflow as tf
import os
import rospy
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.done = 0
        self.out = 0

        # get working directory path
        working_directory = os.path.dirname(os.path.realpath(__file__))

        # graph to detect traffic lights from image
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            gdef = tf.GraphDef()
            with open(working_directory + "/frozen_inference_graph.pb", 'rb') as f:
                gdef.ParseFromString( f.read() )
                tf.import_graph_def( gdef, name="" )

            self.session_dg = tf.Session(graph=self.detection_graph )
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes =  self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        roi = self.roi_for_traffic_light(image)

        if roi is None:
            return TrafficLight.UNKNOWN

        class_image = cv2.resize( image[roi[0]:roi[2], roi[1]:roi[3]], (32,32) )
        if self.out:
            cv2.imwrite('/home/imageration.jpg',class_image)
            self.done=1
        return TrafficLight.UNKNOWN

    def roi_for_traffic_light(self, image):
        with self.detection_graph.as_default():
            #switch from BGR to RGB. Important otherwise detection won't work
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image,axis=0)
            #run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session_dg.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: tf_image_input})

            detection_boxes   = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores  = np.squeeze(detection_scores)


            ret = None
            detection_threshold = 0.4

            # Find first detection of signal. It's labeled with number 10
            idx = -1
            for i, cl in enumerate(detection_classes.tolist()):
                if cl == 10:
                    idx = i;
                    break;

            if idx == -1:
                pass  # no signals detected
            elif detection_scores[idx] < detection_threshold:
                pass # we are not confident of detection
            else:
                dim = image.shape[0:2]
                box = self.from_normalized_dims__to_pixel(detection_boxes[idx], dim)
                box_h, box_w  = (box[2] - box[0], box[3]-box[1])
                if (box_h < 20) or (box_w < 20):
                    rospy.logwarn("Box too small")
                    pass    # box too small
                elif ( box_h/box_w < 1.6):
                    rospy.logwarn("Box wrong ratio: "+str(box))
                    self.out=1
#                    pass    # wrong ratio
                    ret = box
                else:
                    if self.done==1:
                        self.out=0
                    rospy.loginfo('detected bounding box: {} conf: {}'.format(box, detection_scores[idx]))
                    ret = box

        return ret

    def from_normalized_dims__to_pixel(self, box, dim):
            height, width = dim[0], dim[1]
            box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
            return np.array(box_pixel)
