import matplotlib.pyplot as plt

from rfcn.datasets import PascalInstanceSegmentationDataset


if __name__ == '__main__':
    dataset = PascalInstanceSegmentationDataset('val')
    img_viz = dataset.visualize_example(0)
    plt.imshow(img_viz)
    plt.show()
