import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import datetime
import random


def show_images(target_dir, class_name):
    """
    Shows 3 random images from `class_name` folder in `target_dir` folder.

    Parameters
    ----------
    - target_dir: Path to the parent directory that contains class_name folders.
    - class_name: Name of the folder where images of a particular class are present.
    """
    target_path = target_dir + class_name
    file_names = os.listdir(target_path)
    target_images = random.sample(file_names, 3)

    # Plot images
    plt.figure(figsize=(16, 10))
    for i, img in enumerate(target_images):
        img_path = target_path + "/" + img
        plt.subplot(1, 3, i + 1)
        plt.imshow(mpimg.imread(img_path))
        plt.title(class_name)
        plt.axis("off")


def plot_curves(history):
    """
    Plots training, validation accuracy and loss curves.

    Parameters
    ----------
    - history: TensorFlow model History object
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


def preprocess_image(image_path, image_size=224, scale=True):
    """
    Reads the provided image, converts it into a tensor and reshapes into (224, 224, 3).

    Parameters
    ----------
    - image_path: path to the target image
    - image_size: size to resize target image to, default 224
    - scale : whether to scale pixel values to range(0, 1), default True

    Returns
    ----------
    - img: resized and rescaled tensor of shape (224, 224, 3)
    """

    img = tf.io.read_file(image_path)
    # Decode the image into tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [image_size, image_size])
    if scale:
        return img / 255.
    else:
        return img


def make_predictions(model, class_names, image_path):
    """
    Make predictions on the given image, with the provided model and plots the image with the predicted class as the title.

    Parameters
    ----------
    - model: trained CNN model
    - image_path: path to the target image
    - class_names: class names present in the dataset
    """
    # Import the target image and preprocess it
    img = preprocess_image(image_path)

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Get the predicted class
    if len(pred[0]) > 1:  # check for multi-class
        pred_class = class_names[pred.argmax()]  # if more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # if only one output, round

    # Plot the image and predicted class
    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


def create_tensorboard_callback(save_path, experiment_name):
    """
    Creates a TensorBoard callback to store log files.

    Stores log files with the filepath:
      "save_path/experiment_name/current_datetime/"

    Parameters
    ----------
    - save_path: target directory to store TensorBoard log files
    - experiment_name: name of experiment directory

    Returns
    ----------
    - callback: TensorBoard callback
    """
    log_dir = save_path + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_checkpoint_callback(checkpoint_path):
    """
    Creates a ModelCheckpoint callback to store model checkpoint every epoch.

    Stores checkpoints with the filepath:
      "checkpoint_path/"

    Parameters
    ----------
    - checkpoint_path: target directory to save model checkpoints

    Returns
    ----------
    - callback: ModelCheckpoint callback
    """
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             save_best_only=False,
                                                             save_freq="epoch",
                                                             verbose=1)
    return checkpoint_callback


def compare_historys(original_history, new_history, initial_epochs=25):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
