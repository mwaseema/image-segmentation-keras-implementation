import os

from keras.utils import plot_model


def make_model_diagram(model, output_png_path, show_shapes=True, show_layer_names=True):
    # make folder if doesn't exist
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)

    plot_model(
        model,
        to_file=output_png_path,
        show_shapes=show_shapes,
        show_layer_names=show_layer_names,
    )
