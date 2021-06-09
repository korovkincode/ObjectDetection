from pixellib.instance import instance_segmentation

def detection():
	image = instance_segmentation()
	#You can install this model from https://github.com/matterport/Mask_RCNN/releases/
	image.load_model('mask_rcnn_coco.h5')
	target_class = image.select_target_classes(person = True)
	image.segmentImage(show_bboxes = True, segment_target_classes = target_class, image_path = 'city.jpg', output_image_name = 'result.jpg')

detection()