from pixellib.instance import instance_segmentation

def detection():
	image = instance_segmentation()
	#You can install this model from https://github.com/matterport/Mask_RCNN/releases/
	image.load_model('mask_rcnn_coco.h5')
	target_class = image.select_target_classes(person = True)
	res = image.segmentImage(
		show_bboxes = True,
		segment_target_classes = target_class,
		image_path = 'city.jpg',
		output_image_name = 'result.jpg'
	)
	people_cnt = 0
	for score in res[0]['scores']:
		if score >= 0.6:
			people_cnt += 1
	print(people_cnt)

detection()