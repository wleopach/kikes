import matplotlib.pyplot as plt
import seaborn as sns


def plot_hist(atr_name: str, slices :dict, desc: dict):
    slices[atr_name.split('_')[0]][atr_name].hist()
    plt.title(desc[atr_name])
    plt.show()


def plot_join(x, y, labels, save=False, path=None):
    """ Draw a plot of two variables with bivariate and univariate graphs
        x =      x coordinate
        y =      y coordinate
        labels = classes of the points
        save = boolean to decide if save
        path = path to store the image
    """
    _ = sns.jointplot(
        x=x, y=y, kind="kde", hue=labels
    )
    if save:
        if not path:
            path = input("please type path to image with file extension")
        plt.savefig(path)
    plt.show()
