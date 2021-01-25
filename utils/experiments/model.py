import logging

from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.utils.vis_utils import plot_model


def save_model_based_on_thresholds(model, path, epoch, metrics, thresholds):
    ok = True

    for m in thresholds:
        if metrics[m] <= thresholds[m]:
            ok = False

    if ok:
        save_model(model, path, epoch)


def save_model(model, path, epoch):
    model_path = '{}/model.json'.format(path)
    weights_path = '{}/weights_e{}.h5'.format(path, epoch)

    model_json = model.to_json()

    with open(model_path, 'w') as json_file:
        json_file.write(model_json)

    logging.debug('Saved model to disk.')

    # visualize_model(model, path)  # TODO: does not work on server

    model.save_weights(weights_path)
    logging.debug('Saved weights of epoch {} to disk.'.format(epoch))


def load_model(path, epoch):
    model_path = '{}/model.json'.format(path)
    weights_path = '{}/weights_e{}.h5'.format(path, epoch)

    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    logging.debug('Loaded and initialized model from disk.')

    model.load_weights(weights_path)
    logging.debug('Loaded weights of epoch {} from disk into model.'.format(epoch))

    return model


def visualize_model(model, path):
    plot_path = '{}/model.png'.format(path)

    plot_model(model,
               to_file=plot_path,
               show_shapes=True,
               show_layer_names=False,
               expand_nested=False,
               dpi=100)

    logging.debug('Visualized model as PNG file.')