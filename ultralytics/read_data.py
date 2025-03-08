import PIL.Image as Image
import numpy as np

def read_instance_png(path_instance_png):
    img = np.array(Image.open(path_instance_png))
    print('img', img)
    obj_ids = np.unique(img)
    # Create a dictionary with the object instances ids as keys and the class ids as values
    print(obj_ids)
    objects = {obj_id % 1000: obj_id // 1000 for obj_id in obj_ids}
    obj_id = obj_ids[0]
    print(obj_id)

if __name__ == "__main__":
    path_single_image_instance = "/projects/master/c5/KITTI_MOTS/instances/0000/000000.png"
    read_instance_png(path_single_image_instance)
    img = Image.open(path_single_image_instance)

    path_single_image_instance_txt = "/projects/master/c5/KITTI_MOTS/instances_txt/0000/000000.txt"
    with open(path_single_image_instance_txt, "w") as file:
        print("file", file)

    img.show()
    path_single_image_file = "/projects/master/c5/KITTI_MOTS/training/image_02/0000/000000.png"
    print("read_data.py is being run directly")