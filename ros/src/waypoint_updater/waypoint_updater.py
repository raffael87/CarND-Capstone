#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree
import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.pose = None
        self.base_waypoints = None
        self.stopline_wp_index = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(10) # if 50 car does not follow waypoints, possible solution deactivating logic in pure pursuit cpp
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y

        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        # check if it is infront of ego vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        previous_coord = self.waypoints_2d[closest_idx -1]

        # check with three points if closes is in infront
        position_vec = np.array([x,y])
        previous_vec = np.array(previous_coord)
        closest_vec = np.array(closest_coord)

        val = np.dot(closest_vec - previous_vec, position_vec - closest_vec)

        if val > 0: # closest waypoint is behind so we take next one
            closest_idx = (closest_idx + 1) %  len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self):#, closest_idx):
        #lane = Lane()
        #lane.header = self.base_waypoints.header
        #lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_index == -1 or (self.stopline_wp_index >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []

        for i, waypoint in enumerate(waypoints): # check if slice to make efficient
            p = Waypoint() # new message
            p.pose = waypoint.pose

            stop_index = max(self.stopline_wp_index - closest_idx - 2, 0) # stop two waypoints before line, without -2 we would stop with car center on line
            calc_distance = self.distance(waypoints, i, stop_index)
            velocity = math.sqrt(2 * MAX_DECEL * calc_distance) # as distance between waypoint and stop waypoint get small we decelerate

            if velocity < 1.0:
                velocity = 0.0 # stop the car

            p.twist.twist.linear.x = min(velocity, waypoint.twist.twist.linear.x) # because of square root, if distance is too far, square root (velocity) would be too big.
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        self.pose = msg # cars pose

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d: # check that it is initialized before using it!
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
