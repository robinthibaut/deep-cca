from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from dcca.objectives import cca_loss

__all__ = ["create_model"]


def create_model(
        layer_sizes1,
        layer_sizes2,
        input_size1,
        input_size2,
        learning_rate,
        reg_par,
        outdim_size,
        use_all_singular_values,
        dropout=None,
):
    view1_input = Input(shape=(input_size1,), name="view1_input")
    view2_input = Input(shape=(input_size2,), name="view2_input")

    view1_model_layer = _build_mlp_net(
        layer_sizes1, reg_par, view1_input, dropout=dropout
    )
    view2_model_layer = _build_mlp_net(
        layer_sizes2, reg_par, view2_input, dropout=dropout
    )

    merge_layer = Concatenate(name="merge_layer")(
        [view1_model_layer, view2_model_layer]
    )
    model = Model(inputs=[view1_input, view2_input], outputs=merge_layer)

    # opt = RMSprop(learning_rate=learning_rate)
    opt = SGD(learning_rate=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=opt)

    return model


def _build_mlp_net(layer_sizes: list, reg_par: float, view_input_layer, dropout=None):
    layer = view_input_layer

    for l_id, ls in enumerate(layer_sizes):

        if l_id == len(layer_sizes) - 1:
            activation = "linear"
        else:
            activation = "relu"

        if dropout and l_id == len(layer_sizes) - 1:
            layer = Dropout()(layer)

        layer = Dense(ls, activation=activation, kernel_regularizer=l2(reg_par))(layer)
    return layer
