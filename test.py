from torchvision.datasets import CocoDetection

dataset_train = CocoDetection(
    img_folder="dataset/train2017",
    ann_file="dataset/annotations/instances_train2017.json"
)
print(len(dataset_train))  # должно быть >0

dataset_val = CocoDetection(
    img_folder="dataset/val2017",
    ann_file="dataset/annotations/instances_val2017.json"
)
print(len(dataset_val))  # должно быть >0
