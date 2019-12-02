import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import imagehash
from skimage import filters, morphology
from skimage.util import random_noise
from keras.preprocessing.image import ImageDataGenerator
from imblearn.keras import balanced_batch_generator


def plot_frequency_distribution(dataset):
    data = dataset.get_class_distribution()
    
    # Set style
    sns.set_style("white")
    sns.set_style("ticks")
    
    # Colour palette
    sns.color_palette("hls", 10)
    
    # Create new figure
    plt.subplots(figsize=(11, 3))
    # Create plot
    sns.barplot(y='class', x='count', data=data, palette='hls')
    plt.xticks(np.linspace(0, 2250, 10))
    # Remove spine
    sns.despine(bottom=True)
    #  Add grid lines
    plt.grid(b=True, which='major', axis='x')
    # Remove axis labels
    plt.ylabel("")
    plt.xlabel("")
    plt.show()


def image_as_square(image):
    IMAGE_WIDTH = 48
    IMAGE_HEIGHT = IMAGE_WIDTH
    
    return np.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.uint8)


def image_as_array(image):
    IMAGE_WIDTH = 48
    IMAGE_HEIGHT = IMAGE_WIDTH
    return np.resize(image, (IMAGE_WIDTH * IMAGE_HEIGHT,)).astype(np.uint8)


def display_image_before_after(before_image: np.array, after_image: np.array):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    
    ax = axes.ravel()
    
    resized_before = image_as_square(before_image)
    resized_after = image_as_square(after_image)
    
    ax[0].imshow(resized_before, cmap='gray', vmin=0, vmax=255)
    ax[0].axis('off')
    
    ax[1].imshow(resized_after, cmap='gray', vmin=0, vmax=255)
    ax[1].axis('off')
    
    fig.tight_layout()
    plt.show()


def display_images_as_grid(images: np.ndarray):
    image_count = len(images)
    column_count = 8
    fig_width = 11
    
    # Calculate the number of rows needed
    row_count = image_count // column_count
    
    if image_count % column_count > 0:
        row_count += 1
    
    # Calculate the figure size so each image isn't tiny
    fig_height = fig_width / column_count * row_count
    
    # Setup the plot
    fig, axes = plt.subplots(
        nrows=row_count,
        ncols=column_count,
        figsize=(fig_width, fig_height)
    )
    
    ax = axes.ravel()
    
    #  Disable axis on all subplots, even those without images
    for i in range(0, len(ax)):
        ax[i].axis('off')
    
    # Resize images to shape of (48, 48)
    images = np.apply_along_axis(image_as_square, 1, images)
    
    # Plot the images
    for i in range(0, len(images)):
        image = images[i]
        ax[i].imshow(image, cmap='gray', vmin=0, vmax=255)
        ax[i].set_title(i, color='black')
    
    fig.tight_layout()
    plt.show()


def get_image_hash(image_array: np.array):
    image = image_as_square(image_array)
    image = Image.fromarray(image, 'L')
    return imagehash.whash(image)


def compare_image_hashes(hash1, hash2):
    return (hash1 - hash2) / len(hash1.hash) ** 2


def local_contrast(image):
    image = image_as_square(image)
    image = filters.rank.enhance_contrast(image, morphology.disk(2))
    return image_as_array(image)


def add_gaussian_noise(image, *args, **kwargs):
    image = image_as_square(image)
    noisy = random_noise(image, *args, **kwargs)
    noisy = np.multiply(noisy, 255).astype(np.uint8)
    return image_as_array(noisy)


class DataSet:
    labels = {
        '0': 'Speed limit 60',
        '1': 'Speed limit 80',
        '2': 'Speed limit 80 lifter',
        '3': 'Right of way at crossing',
        '4': 'Right of way in general',
        '5': 'Give way',
        '6': 'Stop',
        '7': 'No speed limit general',
        '8': 'Turn right down',
        '9': 'Turn left down'
    }
    
    def __init__(self, type_: str, power_transform=True, balance_data=False):
        self.data_type = type_
        self.is_power_transformed = power_transform
        self.x, self.y = self.read_data()
        if balance_data:
            self.batch_size = 32
            self.__reshape_data_for_keras()
            self.__init_image_data_generator()
            self.__run_balanced_batch_generator()
        
        if power_transform:
            self.power_transform()
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def read_data(self):
        df = pd.read_csv("x_%s_gr_smpl.csv" % self.data_type).astype(int)
        
        # label data-frame rows based on sample data
        for x in range(10):
            index = ~np.loadtxt("y_%s_smpl_%s.csv" % (self.data_type, x),
                                delimiter=",",
                                skiprows=1).astype(bool)  # reversed flags (~)
            df.loc[index, 'label'] = x
        
        x = df.iloc[:, 0:2304].to_numpy()
        y = df.iloc[:, 2304].to_numpy()
        
        # Randomise instance order (forcing the same result each time)
        np.random.seed(42)
        permutation = np.random.permutation(df.shape[0])
        
        x = x[permutation]
        y = y[permutation]
        
        return x, y
    
    def power_transform(self):
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='box-cox')
        pt.fit(self.x)
        self.x = pt.transform(self.x)
    
    def get_class_distribution(self):
        data = pd.DataFrame({
            'numbers': pd.Series([k for k in self.labels.keys()]),
            'class': pd.Categorical([v for v in self.labels.values()]),
            'count': pd.Series([
                np.sum((self.y == 0)),
                np.sum((self.y == 1)),
                np.sum((self.y == 2)),
                np.sum((self.y == 3)),
                np.sum((self.y == 4)),
                np.sum((self.y == 5)),
                np.sum((self.y == 6)),
                np.sum((self.y == 7)),
                np.sum((self.y == 8)),
                np.sum((self.y == 9)),
            ]),
        })
        
        return data
    
    def __reshape_data_for_keras(self):
        image_data = self.x
        image_data = np.apply_along_axis(image_as_square, 1, image_data)
        image_data = image_data.reshape(image_data.shape[0], 48, 48, 1)
        image_data = image_data.astype(np.float32)
        self.x = image_data
    
    def __reshape_data_to_original(self):
        return self.x.reshape(self.x.shape[0], -1)
    
    def __init_image_data_generator(self):
        self.data_generator = ImageDataGenerator(rotation_range=40,
                                                 shear_range=0.2)
        self.data_generator.fit(self.x)
    
    def __run_balanced_batch_generator(self):
        self.generator, self.steps_per_epoch = balanced_batch_generator(
            self.__reshape_data_to_original(), self.y, sampler=RandomOverSampler(), batch_size=self.batch_size,
            keep_sparse=True
        )


from keras.utils.data_utils import Sequence
from imblearn.over_sampling import RandomOverSampler


class BalancedDataGenerator(Sequence):
    
    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = batch_size
        self._shape = x.shape
        datagen.fit(x)
        self.gen, self.steps_per_epoch = balanced_batch_generator(x.reshape(x.shape[0], -1), y,
                                                                  sampler=RandomOverSampler(),
                                                                  batch_size=self.batch_size, keep_sparse=True)
    
    def __len__(self):
        return self._shape[0] // self.batch_size
    
    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()


train_data = DataSet('train', False, True)
subset = train_data.x
x_train = np.apply_along_axis(image_as_square, 1, subset)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype(np.float32)
#
datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
)

# test_image = Image.fromarray(train_data.x[0], 'L')
# input_image = test_image.reshape()
#
a = datagen.flow(x_train)

balanced_gen = BalancedDataGenerator(x_train, train_data.y, datagen, batch_size=32)
steps_per_epoch = balanced_gen.steps_per_epoch

batch_x, batch_y = balanced_gen[0]
batch_x2 = []
for i in range(0, 32):
    batch_x2.append(batch_x[i].reshape(48, 48))

batch_x2 = [image_as_array(i) for i in batch_x2]
batch_x2 = np.array(batch_x2)

display_images_as_grid(batch_x2)

# test_x, test_y = read_data('test')

# class_distribution = train_data.get_class_distribution()
# # plot_frequency_distribution(train_data.y)
#
# test_image, _ = train_data[1]
# test_image_hash = get_image_hash(test_image)
#
# modified_image = test_image
# modified_image = local_contrast(modified_image)
# modified_image = add_gaussian_noise(modified_image, mode='poisson')
# modified_image_hash = get_image_hash(modified_image)
#
# hash_difference = compare_image_hashes(test_image_hash, modified_image_hash)
# print(f"Hash difference: {hash_difference}")

# display_images_as_grid(train_data.x[0:200])

# display_image_before_after(test_image, modified_image)
