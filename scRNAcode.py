#imports
import numpy as np
import umap
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score

#heatmap function from matplotlib
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

#Load the data
X = np.load("drive/MyDrive/Colab Notebooks/scRNA_full.npy")

#Open labels
f = open("drive/MyDrive/Colab Notebooks/label.txt")
output = f.readlines()
output = [item.rstrip() for item in output]

#Get unique labels and assign them to a list
uniqueLabels = np.unique(output)

#Assign all original labels with the index in the unique labels list
y_hcL = [list(uniqueLabels).index(item) for item in output]
y_hc = np.array(y_hcL)

#Split the data
X_train, X_test,Y_train, Y_test = train_test_split(X, y_hc, test_size=1/3.0, random_state=0)

#Model Architecture
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu", name = "L1"),
    layers.Dense(128, activation="relu", name = "L2"),
    layers.Dense(14, activation="softmax")
])
input_ = np.random.normal(size=(1, 13822))
_ = model(input_)
print(model.summary())

#Compile the model
model.compile(optimizer=tf.keras.optimizers.experimental.SGD(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#Fit the model
history = model.fit(X_train, Y_train, epochs=20, batch_size= 32, validation_data=(X_test,Y_test))

#Plot accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('drive/MyDrive/Colab Notebooks/accuracy_4layers.png',dpi=300)
plt.show()

#Plot loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('drive/MyDrive/Colab Notebooks/loss_4layers.png',dpi=300)
plt.show()

#Feature Extraction
feature_extractor = keras.Model(
   inputs=model.inputs,
   outputs=model.get_layer(name="L2").output,
)
x = X
X_feature = feature_extractor(x)

#Test the model
test_digits = X_test
predictions = model.predict(test_digits)
y_predict = np.argmax(predictions, axis=1)

#Dimension reduction for plotting
umap=umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation')
X_umap=umap.fit_transform(X_feature)

#Plot the results
colorList = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', '#d6730f', '#9f0dde', '#ed8ec4', '#b7de85', '#c498eb', '#19e3a6', '#79767a', '#0d0c0d']
for i in range(14):
  plt.scatter(X_umap[y_hc == i, 0], X_umap[y_hc == i, 1], color= colorList[i], s=15, label=uniqueLabels[i], edgecolors='black')
plt.legend(bbox_to_anchor=(1, 1), loc= 'upper left')
plt.title('Hierarchical Clustering')
plt.ylabel('dim 1')
plt.xlabel('dim 2')
plt.show()

#Confusion matrix
cm = confusion_matrix(Y_test, y_predict)
print(cm)
fig, ax = plt.subplots()
vegetables = ['CD56 (bright) NK cells', 'CD56 (dim) NK cells', 'MAIT T cells',
       'classical monocytes', 'effector CD8 T cells',
       'intermediate monocytes', 'memory B cells', 'memory CD4 T cells',
       'myeloid DC', 'naive B cells', 'naive CD4 T cells',
       'naive CD8 T cells', 'non-classical monocytes', 'plasmacytoid DC']
farmers = np.arange(1,15)
im, cbar = heatmap(cm, vegetables, farmers, ax=ax,
                   cmap="YlGn", cbarlabel="")
texts = annotate_heatmap(im, valfmt='{x:,g}', size=7)
fig.tight_layout()
plt.savefig('drive/MyDrive/Colab Notebooks/cmHeatmap.png',dpi=300)
plt.show()

#Accuracy, precision, recall, f1 score
ac_s = accuracy_score(Y_test, y_predict)
f1_s = f1_score(Y_test, y_predict, average = 'micro')
p_s = precision_score(Y_test, y_predict,average = 'micro')
r_s = recall_score(Y_test, y_predict,average = 'micro')
print(ac_s, f1_s, p_s, r_s)