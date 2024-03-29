from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, histogram_freq=None, write_graph=True, write_grads=False, write_images=False, update_freq=4096):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, histogram_freq=histogram_freq, write_graph=write_graph, write_grads=write_grads, write_images=write_images, update_freq=update_freq)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.learning_rate)})
        super().on_epoch_end(epoch, logs)
