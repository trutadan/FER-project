from bounding_box.FacesBoundingBox import FacesBoundingBox

if __name__ == '__main__':
    dataset_path = "../../dataset_kdef"
    dataset_name = "kdef"
    image_format = ".jpg"

    faces_bounding_boxes = FacesBoundingBox(dataset_path, dataset_name, image_format)

    faces_bounding_boxes.process()
    faces_bounding_boxes.save()

    print(faces_bounding_boxes)
