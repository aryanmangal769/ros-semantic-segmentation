#!/usr/bin/env python
from __future__ import print_function

import os

import cv2
import numpy as np
import cv_bridge

import ugv_bot.srv
from ugv_bot.srv import SendImage, SendImageResponse
import rospy
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError
import cv_bridge


def rgba2rgb( rgba, background=(255,255,255) ):
	row, col, ch = rgba.shape

	if ch == 3:
		print("Yess")
		return rgba

	assert ch == 4, 'RGBA image has 4 channels.'

	rgb = np.zeros( (row, col, 3), dtype='float32' )
	r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

	a = np.asarray( a, dtype='float32' ) / 255.0

	R, G, B = background

	rgb[:,:,0] = r * a + (1.0 - a) * R
	rgb[:,:,1] = g * a + (1.0 - a) * G
	rgb[:,:,2] = b * a + (1.0 - a) * B

	return np.asarray( rgb, dtype='uint8' )


def handle_image(request):
	global img
	if request:

		img = rgba2rgb(img)


		srv_msg = SendImageResponse()
		srv_msg.header.stamp = rospy.Time.now()
		srv_msg.format = "jpeg"
		srv_msg.data = np.array(cv2.imencode('.jpg',img)[1]).tostring()

		print("Hey Homie........")
		print("Done Sending !!!!")
		print("===============================================")

		return srv_msg
def Img_msg_Publisher(final_img):

  #------------------Publish Final Image to rviz-----------------------

  image_pub = rospy.Publisher("/ugv_bot/image_raw",Image, queue_size = 1)

  # bridge=CvBridge()
  image_msg = cv_bridge.cv2_to_imgmsg(final_img,encoding='bgr8') #, 
  image_pub.publish(image_msg)


if __name__ == "__main__":
	# global img
# global cap
	cap = cv2.VideoCapture('/home/aryan/Downloads/2016_0101_000538_011.MP4')
	rospy.init_node('video_server')
	if (cap.isOpened()== False): 
		print("Error opening video stream or file")

	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, img = cap.read()
		if ret == True:
			# image_server()   

			cv2.imshow('Frame',img)
			Img_msg_Publisher(img)
			# Press Q on keyboard to  exit
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

		# Break the loop
		else: 
			break

	# When everything done, release the video capture object
	cap.release()

	# Closes all the frames
	cv2.destroyAllWindows()
	



