# def yolov3_inference_image(config, dataset_name, inference_image, log):
    
#     log.info("Initializing model...", task="inference")
#     model     = YOLOv3(device=config["device"], num_classes=len(config["datasets"][dataset_name]["classes"])).to(config["device"])

#     log.info("Initializing optimizer...", task="inference")
#     optimizer = optim.Adam(model.parameters(), lr=config["dl"]["yolov3"]["learning_rate"], weight_decay=config["dl"]["yolov3"]["weight_decay"]) 
    
#     log.info("Load existing weights...", task="inference")
#     yolov3_load_checkpoint(config, config["datasets"][dataset_name]["save_checkpoint_file"], model, optimizer, config["dl"]["yolov3"]["learning_rate"])

#     log.info("Loading inference dataset & dataloader...", task="inference")
#     inference_loader = yolov3_loaders(config, dataset_name, "unseen", log)

#     log.info("Predicted bboxes...", task="inference")
#     pred_boxes, images_pred_bboxes = yolov3_get_inference_bboxes(inference_loader, 
#                                                                  model, 
#                                                                  iou_threshold=config["dl"]["yolov3"]["nms_iou_threshold"], 
#                                                                  anchors=config["dl"]["yolov3"]["anchors"],
#                                                                  threshold=config["dl"]["yolov3"]["confidence_threshold"],
#                                                                  box_format=config["dl"]["yolov3"]["box_format"],
#                                                                  device=config["device"])

#     log.info("Plotting bboxes...", task="inference")
#     cv2im = cv2.imread(inference_image)
#     height, width, _    = cv2im.shape
#     if inference_image.endswith((".png")):
#         inference_image_bbox = inference_image.replace(".png", "_bboxes.png")
#     if inference_image.endswith((".jpg")):
#         inference_image_bbox = inference_image.replace(".jpg", "_bboxes.png")
#     if inference_image.endswith((".tif")):
#         inference_image_bbox = inference_image.replace(".tif", "_bboxes.png")
#     for image, bboxes in images_pred_bboxes.items():

#         # Create a Rectangle patch
#         for bbox in bboxes:
#             assert len(bbox) == 7, "box should contain image index, class pred, confidence, x, y, width, height"
#             class_pred      = bbox[1]
#             class_prob      = round(bbox[2]*100, 2)
#             bbox            = bbox[3:]
#             # After slicing
#             # box[0] is x midpoint, box[1] is y midpoint, box[2] is width, box[3] is height
#             upper_left_x    = bbox[0] - bbox[2] / 2
#             upper_left_y    = bbox[1] - bbox[3] / 2
#             lower_right_x   = bbox[0] + bbox[2] / 2
#             lower_right_y   = bbox[1] + bbox[3] / 2
#             upper_left_x    = int(upper_left_x * width)
#             upper_left_y    = int(upper_left_y * height)
#             lower_right_x   = int(lower_right_x * width)
#             lower_right_y   = int(lower_right_y * height)
#             bbox_width      = int(bbox[2] * width)
#             bbox_height     = int(bbox[3] * height)

#             local_bbox = {
#                 "bbox_ul": (upper_left_x, upper_left_y),
#                 "bbox_br": (lower_right_x, lower_right_y)
#             }

#             if int(class_prob)>75:
#                 cv2im = cv2.rectangle(cv2im, local_bbox["bbox_ul"], local_bbox["bbox_br"], config["opencv"]["bbox_color"], config["opencv"]["bbox_thickness"])

#     cv2.imwrite(inference_image_bbox, cv2im)
    
#     log.info("Completed inference...", task="inference")
