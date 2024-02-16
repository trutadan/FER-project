from bounding_box.FacesBoundingBox import FacesBoundingBox

if __name__ == '__main__':
    dataset_path = "../../dataset_ck+"
    dataset_name = "ck+"
    image_format = ".png"

    faces_bounding_boxes = FacesBoundingBox(dataset_path, dataset_name, image_format)

    faces_bounding_boxes.process()
    faces_bounding_boxes.save()

    print(faces_bounding_boxes)
