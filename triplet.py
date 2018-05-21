class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        a, p, n = inputs
        p_dist = K.sum(K.square(a-p), axis=-1)
        n_dist = K.sum(K.square(a-n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def build_model():
    base_model_input = Input((259,), name='base_input')
    base_model_layer1 = Dense(32)(base_model_input)
    base_model_layer2 = Dense(2)(base_model_layer1)
    base_model = Model(base_model_input,base_model_layer2,name='base_model')
    positive_item_input = Input((259,), name='positive_item_input')
    negative_item_input = Input((259,), name='negative_item_input')
    anchor_item_input = Input((259,), name='anchor_item_input')
    positive_item_embed = base_model(positive_item_input)
    negative_item_embed = base_model(negative_item_input)
    anchor_item_embed = base_model(anchor_item_input)
    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([anchor_item_embed, positive_item_embed, negative_item_embed])
    model = Model(input=[anchor_item_input,positive_item_input,negative_item_input],
                  output=triplet_loss_layer)
    model.compile(loss=identity_loss, optimizer=Adam(lr=0.1,clipnorm=10.0, decay=0.1))
