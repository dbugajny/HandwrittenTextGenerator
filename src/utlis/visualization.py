import matplotlib.pyplot as plt


def show_image(image, title=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()
