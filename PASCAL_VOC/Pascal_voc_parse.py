import xml.etree.ElementTree as ET
import cv2
import numpy
import os


#Class for reading names and box locations from xml file
def read_content(xml_file: str):

    tree = ET.parse('/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/Annotations/' + xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    objname_list = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text
        
        objname = str(boxes.find("name").text)
        objname_list.append(objname)

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    print(objname_list)
    print(list_with_all_boxes)
    
    return objname_list, list_with_all_boxes



#Class for extracting actual subimages (and resizing)
def extract_subimage(img_file: str, names, boxes, out_path, subimg_cnt):
	image = cv2.imread(img_file)
	original = image.copy()
	dim=(224,224)
	#loop through all objects found in the xml file during read_content
	for idx, name in enumerate(names):
		x,y,w,h = boxes[idx]
		
		#crop using xml box and then resize to 224X224
		subimg = cv2.resize(original[y:h, x:w], dim)
		
		folder = out_path + name
		#create label folder in "Results" directory if one does not exist
		isExist = os.path.exists(folder)
		if not isExist:
			os.makedirs(folder)
		
		cv2.imwrite(os.path.join(folder ,'subimg_{}.jpg'.format(subimg_cnt)), subimg)
		cv2.waitKey(0)
		subimg_cnt += 1
		
		#For manual debugging, allows to go through subimages one by one
		#cv2.imshow('', subimg)
		#print('SUBIMAGE LABEL', name)
		#cv2.waitKey()

# main function for looping through annotation folder	
def main():
	#outputting to user defined "Results" folder
	outPath = "/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/Results/"
	an_path = "/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/Annotations/"
	im_path = "/home/ali/anaconda3/lib/python3.8/VOCdevkit/VOC2012/JPEGImages/"
	subimg_cnt = 0
	for annotation_path in os.listdir(an_path):
		#extract filename from xml for use in img folder
		im_file = im_path + annotation_path.replace('.xml','.jpg')
		
		names, boxes = read_content(annotation_path)
		extract_subimage(im_file, names, boxes, outPath, subimg_cnt)


main()

