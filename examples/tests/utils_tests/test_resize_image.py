import matplotlib.pyplot as plt

import skimage.color

import rfcn


if __name__ == '__main__':
    dataset = rfcn.datasets.PascalInstanceSegmentationDataset('val')
    datum, label_class, label_instance = dataset.get_example(0)
    img = dataset.datum_to_img(datum)

    height, width = img.shape[:2]
    height_32s, width_32s = height // 32, width // 32

    img_32s = rfcn.utils.resize_image(img, (height_32s, width_32s))
    label_instance_32s = rfcn.utils.resize_image(
        label_instance, (height_32s, width_32s))
    label_viz = skimage.color.label2rgb(label_instance_32s, img_32s)

    plt.subplot(121)
    plt.imshow(img_32s)
    plt.subplot(122)
    plt.imshow(label_viz)
    plt.show()
