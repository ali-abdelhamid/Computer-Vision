import xml.etree.ElementTree as ET
import cv2
import numpy
import os

#File to run on results folder after running Pascal_voc_parse.py file on the dataset
def main():
	outPath = "/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/"
	main_path = "/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/Results/"
	im_path = "/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/JPEGImages/"
	list_major = numpy.empty((1,50177))
	list_major_m = numpy.empty((1,50177))
	for path in os.listdir(main_path):
		for img in os.listdir(main_path + path):
			loc = main_path + path + '/' + img
			image = cv2.imread(loc, 0)
			
			a = numpy.asarray(image)
			list_minor = a.flatten()
			list_minor = numpy.append(list_minor, path)
			list_major = numpy.vstack([list_major, list_minor])
	list_major_m = numpy.vstack([list_major_m, list_major])
	print(list_major_m.shape)
	
	numpy.savetxt(os.path.join(outPath, 'binary_file.csv'), list_major, delimiter=',', fmt='%s')

main()
