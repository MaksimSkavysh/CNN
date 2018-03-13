import tflearn
import numpy as np
import matplotlib.pyplot as plt
import six


def plot_figure(result, filename=''):
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()


def display_convolutions(model, layer, padding=4, filename=''):
    if isinstance(layer, six.string_types):
        vars = tflearn.get_layer_variables_by_name(layer)
        vars = tflearn.get_layer_variables_by_name(layer)
        variable = vars[0]
    else:
        variable = layer.W

    data = model.get_weights(variable)

    filter_size = data.shape[0]
    filter_depth = data.shape[2]
    number_of_filters = data.shape[3]
    N = filter_depth * number_of_filters  # N is the total number of convolutions

    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(number_of_filters)))
    result_size = filters_per_row * (filter_size + padding) - padding

    result = np.zeros((result_size, result_size, 4))
    filter_x = 0
    filter_y = 0
    for filter_number in range(number_of_filters):
        if filter_x == filters_per_row:
            filter_y += 1
            filter_x = 0
        for i in range(filter_size):
            for j in range(filter_size):
                plot_i = filter_y * (filter_size + padding) + i
                plot_j = filter_x * (filter_size + padding) + j

                result[plot_i, plot_j, 0] = data[i, j, 0, filter_number]
                result[plot_i, plot_j, 1] = data[i, j, 1, filter_number]
                result[plot_i, plot_j, 2] = data[i, j, 2, filter_number]
                result[plot_i, plot_j, 3] = data[i, j, 3, filter_number]
        filter_x += 1

    # Normalize image to 0-1
    min, max = result.min(), result.max()
    result = (result - min) / (max - min)

    plot_figure(result, filename)
