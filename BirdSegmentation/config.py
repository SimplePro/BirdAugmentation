import os

# (key)_(value)_(type) 형식으로 변수명을 지었다. -> (key)_(value)_(type)[(key)] = (value)
class_imgid_list = [[] for _ in range(201)] # class_imgid_list[class] = [imgid1, imgid2 ... imgidN]
class_name_list = [-1] # class_name_list[class] = class_name
imgid_filename_dict = {} # imgid_filename_dict[imgid] = filename
imgid_bbox_dict = {} # imgid_bbox_dict[imgid] = (xmin, ymin, w, h)

root_dir = os.path.join("/", "home", "kdhsimplepro", "kdhsimplepro", "AI", "BirdAugmentation")

# class_imgid_list
with open(os.path.join(root_dir, "CUB200", "image_class_labels.txt"), "r") as f:
    for line in f.readlines():
        imgid, class_ = map(int, line.split())

        class_imgid_list[class_].append(imgid)

# class_name_list
with open(os.path.join(root_dir, "CUB200", "classes.txt"), "r") as f:
    for line in f.readlines():
        class_name_list.append(line.split()[1])

# imgid_filename_dict
with open(os.path.join(root_dir, "CUB200", "images.txt"), "r") as f:

    for line in f.readlines():
        imgid, file_name = line.split()
        file_name = file_name.split("/")[1]
        imgid_filename_dict[int(imgid)] = file_name

# imgid_bbox_dict
with open(os.path.join(root_dir, "CUB200", "bounding_boxes.txt"), "r") as f:
    for line in f.readlines():
        imgid, xmin, ymin, w, h = map(float, line.split())

        imgid_bbox_dict[int(imgid)] = (xmin, ymin, w, h)