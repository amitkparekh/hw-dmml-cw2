import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, freeze_support

class DataSet:
    """

    Methods
    -------
    read_data()
    get_class_distribution()
    extend_data()
    mutate()
    power_transform()
    export()
    display_image(index)
    display_image_grid([lower_index, upper_index])
    """
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

    image_width = 48
    image_height = 48

    def __init__(self,
                 type_: str,
                 power_transform=False,
                 mutate=False,
                 extend_data=False):

        self.data_type = type_
        self.x, self.y = self.read_data()

        self.dist = self.get_class_distribution()

        if extend_data:
            print(f"Extending {self.data_type} data...")
            self.extend_data()

        if mutate and not power_transform:
            print(f"Mutating {self.data_type} data...")
            self.mutate()

        if power_transform and not mutate:
            print(f"Transforming {self.data_type} data...")
            self.power_transform()

        if power_transform and mutate:
            raise Exception(
                "Can only run EITHER power_transform or mutate. NOT BOTH.")

        print(f"Data imported for {self.data_type}.")

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
        pt = PowerTransformer(method='box-cox', standardize=False)
        pt.fit(self.x)
        self.x = pt.transform(self.x)

    def image_mutation(self, image):
        from skimage import filters, morphology

        image = self.__image_as_square(image)

        # Enhance the image by overtly enhancing the contrast.
        image = filters.rank.enhance_contrast(image, morphology.disk(2))
        image = filters.rank.autolevel(image, morphology.disk(4))

        # Resize the image back to a shape of (2304, )
        return self.__image_as_array(image)

    def mutate(self):
        self.x = np.apply_along_axis(self.image_mutation, 1, self.x)

    def export(self, filename_prefix: str):
        import os
        filename = f"{filename_prefix}-{self.data_type}"

        np.savetxt(f"{filename}-x.csv", self.x, delimiter="\n")
        print(f"[EXPORT] Exported x data to: {os.getcwd()}/{filename}-x.csv")

        np.savetxt(f"{filename}-y.csv", self.y, delimiter="\n")
        print(f"[EXPORT] Exported y data to: {os.getcwd()}/{filename}-y.csv")

    def get_class_distribution(self):
        data = pd.DataFrame({
            'label':
            pd.Series([k for k in self.labels.keys()]),
            'class':
            pd.Categorical([v for v in self.labels.values()]),
            'count':
            pd.Series([
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

        data['need'] = data['count'].max() - data['count']

        return data

    def plot_class_distribution(self):
        data = self.get_class_distribution()

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

    def __image_as_square(self, image):
        return np.resize(image, (self.image_width, self.image_height)).astype(
            np.uint8)

    def __image_as_array(self, image):
        return np.resize(
            image, (self.image_width * self.image_height, )).astype(np.uint8)

    @staticmethod
    def __rescale_image(image):
        return np.interp(image, (np.min(image), np.max(image)), (0, 255))

    def display_image(self, index):

        image = self.x[index]
        image = self.__rescale_image(image)
        image = self.__image_as_square(image)

        # Display the image
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def display_image_grid(self, indexes):

        if len(indexes) != 2:
            raise Exception(
                "Indexes must only have 2 elements: one for lower index, and one for upper."
            )

        images = self.x[indexes[0]:indexes[1]]

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
        fig, axes = plt.subplots(nrows=row_count,
                                 ncols=column_count,
                                 figsize=(fig_width, fig_height))

        ax = axes.ravel()

        #  Disable axis on all subplots, even those without images
        for i in range(0, len(ax)):
            ax[i].axis('off')

        images = np.apply_along_axis(self.__rescale_image, 1, images)
        images = np.apply_along_axis(self.__image_as_square, 1, images)

        # Plot the images
        for i in range(0, len(images)):
            image = images[i]
            ax[i].imshow(image, cmap='gray', vmin=0, vmax=255)
            ax[i].set_title(i, color='black')

        fig.tight_layout()
        plt.show()

    def __reshape_data_for_keras(self, x):
        image_data = x
        image_data = np.apply_along_axis(self.__image_as_square, 1, image_data)
        image_data = image_data.reshape(image_data.shape[0], self.image_width,
                                        self.image_height, 1)
        image_data = image_data.astype(np.float32)
        return image_data

    @staticmethod
    def __reshape_data_from_keras(x):
        return x.reshape(x.shape[0], -1)

    def __get_images_for_label(self, label: int):
        indexes = np.where(self.y == label)
        indexes = np.ravel(indexes).tolist()
        return self.x[indexes]

    @staticmethod
    def __create_image_generator(x):
        from keras.preprocessing.image import ImageDataGenerator
        data_generator = ImageDataGenerator(rotation_range=5,
                                            shear_range=0.2,
                                            height_shift_range=0.05,
                                            width_shift_range=0.05,
                                            zoom_range=[0.99, 1.01])
        data_generator.fit(x)
        return data_generator

    @staticmethod
    def __return_image_batch(data_generator, x_batch, y_batch, batch_size):
        return data_generator.flow(x_batch, y_batch,
                                   batch_size=batch_size).next()

    def __generate_images(self, label, batch_size):
        x = self.__get_images_for_label(label)
        x = self.__reshape_data_for_keras(x)
        y = np.repeat(label, x.shape[0])
        data_generator = self.__create_image_generator(x)
        new_x, new_y = self.__return_image_batch(data_generator, x, y,
                                                 batch_size)
        new_x = self.__reshape_data_from_keras(new_x)
        new_x = new_x.astype(np.uint8)
        return new_x, new_y

    def __get_needed_count(self, label):
        images_needed = self.dist[self.dist['label'] == str(label)]['need']
        return int(images_needed)

    def __update_data_set(self, new_x, new_y):
        for images in new_x:
            self.x = np.vstack((self.x, images))

        for labels in new_y:
            self.y = np.concatenate((self.y, labels))

    def __is_data_imbalanced(self) -> bool:
        if np.sum(self.dist['need'] > 0) > 0:
            return True

        return False

    def extend_data(self):
        new_x_values = []
        new_y_values = []

        if self.__is_data_imbalanced():
            for label in self.dist['label']:
                label = int(label)

                batch_size = self.__get_needed_count(label)

                while batch_size > 0:
                    print(f"[EXTEND] Label {label}: need {batch_size}")

                    new_x, new_y = self.__generate_images(label, batch_size)
                    new_x_values.append(new_x)
                    new_y_values.append(new_y)
                    batch_size = batch_size - len(new_x)

                if batch_size == 0:
                    print(f"[EXTEND] Label {label}: need {batch_size}")
                    continue

            self.__update_data_set(new_x_values, new_y_values)
            self.dist = self.get_class_distribution()

# Keep them global so that all the functions have access to them
train = DataSet('train', power_transform=True, mutate=False, extend_data=True)
test = DataSet('test', power_transform=True, mutate=False, extend_data=True)
train_x = train.x
train_y = train.y
test_x = test.x
test_y = test.y

def get_auc_score(y_pred, test_y):
  uniqueValues, occurCount = np.unique(test_y, return_counts=True)
  y_pred = label_binarize(y_pred, classes=range(0, 10)) # the range here depicts the actual list of classes 
  test_y = label_binarize(test_y, classes=range(0, 10))
  tpr = {}
  fpr = {}
  auc_val = {}
  for i in range(0, 10):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_pred[:, i])
    auc_val[i] = auc(fpr[i], tpr[i])
  auc_values = [x[1] for x in auc_val.items()]
  auc_values = np.asarray(auc_values)
  weighted_product = np.multiply(occurCount, auc_values)
  return float(weighted_product.sum())/float(occurCount.sum())

def k_fold_experiments(clf, train_x, train_y, test_x, test_y):
    k = 10 # as per requirements
    x = np.append(train_x, test_x, axis=0)
    y = np.append(train_y, test_y, axis=0)
    kf = KFold(n_splits=k, random_state=42, shuffle=True)
    auc_scores = []
    f1_scores = []
    precisions =[]
    recalls = []
    for train_index, test_index in kf.split(x,y):
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
      clf.fit(x_train, y_train)
      y_pred = clf.predict(x_test)
      auc_scores.append(get_auc_score(y_pred, y_test))
      f1_scores.append(metrics.f1_score(y_test, y_pred, average='weighted'))
      precisions.append(metrics.precision_score(y_test, y_pred, average='weighted'))
      recalls.append(metrics.recall_score(y_test, y_pred, average='weighted'))
    
    auc_scores = np.asarray(auc_scores)
    f1_scores = np.asarray(f1_scores)
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)

    return (f1_scores.mean(), precisions.mean(), recalls.mean(), auc_scores.mean())

def standard_test_train(clf, train_x, train_y, test_x, test_y):
  clf.fit(train_x, train_y)
  y_pred = clf.predict(test_x)
  return (metrics.f1_score(test_y, y_pred, average='weighted'), metrics.precision_score(test_y, y_pred, average='weighted'), metrics.recall_score(test_y, y_pred, average='weighted'), get_auc_score(y_pred, test_y))

def test_train_with_4000_reduction (clf, train_x, train_y, test_x, test_y):
    idx = np.random.randint(0, len(train_x), 4000)
    c_train_x = np.delete(train_x, idx, axis=0)
    c_train_y = np.delete(train_y, idx, axis=0)
    c_test_x = np.concatenate((test_x, train_x[idx]), axis=0)
    c_test_y = np.concatenate((test_y, train_y[idx]), axis=0)
    return standard_test_train(clf, c_train_x , c_train_y, c_test_x, c_test_y)

def test_train_with_9000_reduction(clf, train_x, train_y, test_x, test_y):
    idx = np.random.randint(0, len(train_x), 9000)
    c_train_x = np.delete(train_x, idx, axis=0)
    c_train_y = np.delete(train_y, idx, axis=0)
    c_test_x = np.concatenate((test_x, train_x[idx]), axis=0)
    c_test_y = np.concatenate((test_y, train_y[idx]), axis=0)
    return standard_test_train(clf, c_train_x , c_train_y, c_test_x, c_test_y)

def run_experiments(clf):
  value_dict = {}
  (value_dict['A-f1-score'], value_dict['A-precision'], value_dict['A-recall'], value_dict['A-AUC']) = k_fold_experiments(clf, train_x, train_y, test_x, test_y)
  (value_dict['B-f1-score'], value_dict['B-precision'], value_dict['B-recall'], value_dict['B-AUC']) = standard_test_train(clf, train_x, train_y, test_x, test_y)
  (value_dict['C-f1-score'], value_dict['C-precision'], value_dict['C-recall'], value_dict['C-AUC']) = test_train_with_4000_reduction (clf, train_x, train_y, test_x, test_y)
  (value_dict['D-f1-score'], value_dict['D-precision'], value_dict['D-recall'], value_dict['D-AUC']) = test_train_with_9000_reduction (clf, train_x, train_y, test_x, test_y)
  return value_dict


if __name__ == "__main__":
    freeze_support()
    rf0 = RandomForestClassifier()
    rf1 = RandomForestClassifier(n_estimators=20)
    rf2 = RandomForestClassifier(n_estimators=50)
    rf3 = RandomForestClassifier(criterion='entropy')
    rf4 = RandomForestClassifier(n_estimators=20, criterion='entropy')
    rf5 = RandomForestClassifier(n_estimators=50, criterion='entropy')
    rf6 = RandomForestClassifier(max_depth=3)
    rf7 = RandomForestClassifier(max_depth=3, criterion='entropy')
    rf8 = RandomForestClassifier(max_depth=10)
    rf9 = RandomForestClassifier(max_depth=10, criterion='entropy')
    rf10 = RandomForestClassifier(min_samples_split=10)
    rf11 = RandomForestClassifier(min_samples_split=10, criterion='entropy')
    classifier_dict = {'Default-Gini-RF': rf0, 'Gini-RF-20-estimators': rf1,
                    'Gini-RF-50-estimators': rf2, 'Default-entropy-RF': rf3,
                    'entropy-RF-20-estimators':rf4, 'entropy-RF-50-estimators': rf5,
                    'Gini-RF-max-depth-3': rf6, 'entropy-RF-max-depth-3': rf7,
                    'Gini-RF-max-depth-10': rf8, 'entropty-RF-max-depth-10': rf9,
                    'Gini-RF-min-split-10': rf10, 'entropy-RF-min-split-10': rf11}
    experiment_names = classifier_dict.keys()
    p = Pool()
    experiment_outcomes = p.map(run_experiments, classifier_dict.values())
    metric_dict = {x[0]:x[1] for x in zip(experiment_names, experiment_outcomes)}
    print(metric_dict)