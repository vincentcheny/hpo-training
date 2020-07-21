import logging
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten, MaxPool2D)
from tensorflow.keras.optimizers import Adam
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
import shutil
import os
import time


_logger = logging.getLogger('cifar10_example')
_logger.setLevel(logging.INFO)


class MnistModel(Model):
    """
    LeNet-5 Model with customizable hyper-parameters
    """
    def __init__(self, filter1,filter2):
        """
        Initialize hyper-parameters.
        Parameters
        ----------
        conv_size : int
            Kernel size of convolutional layers.
        hidden_size : int
            Dimensionality of last hidden layer.
        dropout_rate : float
            Dropout rate between two fully connected (dense) layers, to prevent co-adaptation.
        """
        super().__init__()
        self.conv1 = Conv2D(filters=filter1, kernel_size=(9,9), activation='relu')
        self.pool1 = MaxPool2D(pool_size=2)
        self.conv2 = Conv2D(filters=filter2, kernel_size=(5,5), activation='relu')
        self.pool2 = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.fc1 = Dense(units=50, activation='relu')
        # self.dropout = Dropout(rate=0.1)
        self.fc2 = Dense(units=10, activation='softmax')

    def call(self, x):
        """Override ``Model.call`` to build LeNet-5 model."""
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        # x = self.dropout(x)
        return self.fc2(x)






class ReportIntermediates(Callback):
    """
    Callback class for reporting intermediate accuracy metrics.
    This callback sends accuracy to NNI framework every 100 steps,
    so you can view the learning curve on web UI.
    If an assessor is configured in experiment's YAML file,
    it will use these metrics for early stopping.
    """
    def on_epoch_end(self, epoch, logs=None):
        """Reports intermediate accuracy to NNI framework"""
        # TensorFlow 2.0 API reference claims the key is `val_acc`, but in fact it's `val_accuracy`
        if 'val_acc' in logs:
            # nni.report_intermediate_result(logs['val_acc'])
            print(logs['val_acc'])
        else:
            # nni.report_intermediate_result(logs['val_accuracy'])
            print(logs['val_accuracy'])


def load_dataset():
    """Download and reformat MNIST dataset"""
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # x_train = x_train[..., tf.newaxis]
    # x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


def runtime_eval(x):
    """
    Main program:
      - Build network
      - Prepare dataset
      - Train the model
      - Report accuracy to tuner
    """
    model = MnistModel(filter1=x[2],filter2=x[3])

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=x[0])


    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    (x_train, y_train), (x_test, y_test) = load_dataset()
    
    sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        inter_op_parallelism_threads=int(x[4]),
        intra_op_parallelism_threads=int(x[5]),
        graph_options=tf.compat.v1.GraphOptions(
            build_cost_model=int(x[6]),
            infer_shapes=x[7],
            place_pruned_graph=x[8],
            enable_bfloat16_sendrecv=x[9],
            optimizer_options=tf.compat.v1.OptimizerOptions(
                do_common_subexpression_elimination=x[10],
                max_folded_constant_in_bytes=int(x[11]),
                do_function_inlining=x[12],
                global_jit_level=x[13]))))
    tf.compat.v1.keras.backend.set_session(sess)

    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=x[1],
        epochs=epoch_num,
        verbose=2,
        callbacks=[ReportIntermediates()],
        validation_data=(x_test, y_test)
    )
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    end_time = time.time()

    global final_acc
    final_acc = accuracy
    print("time : %.4f" % float(end_time-start_time))
    return float(start_time - end_time)


def acc_eval(x):
    global final_acc
    print("acc: %.4f" % final_acc)
    return final_acc


final_acc = 0.0


#model para
epoch_num = 81
LR_list = [1e-06, 0.00020506122448979593, 0.0004091224489795919, 0.0006131836734693878, 0.0008172448979591837, 0.0010213061224489796, 0.0012253673469387754, 0.0014294285714285715, 0.0016334897959183674, 0.0018375510204081632, 0.0020416122448979595, 0.0022456734693877553, 0.002449734693877551, 0.002653795918367347, 0.0028578571428571433, 0.003061918367346939, 0.003265979591836735, 0.003470040816326531, 0.0036741020408163267, 0.003878163265306123, 0.004082224489795919, 0.004286285714285715, 0.0044903469387755105, 0.004694408163265306, 0.004898469387755102, 0.005102530612244898, 0.005306591836734694, 0.005510653061224491, 0.0057147142857142865, 0.005918775510204082, 0.006122836734693878, 0.006326897959183674, 0.00653095918367347, 0.006735020408163266, 0.006939081632653062, 0.0071431428571428575, 0.007347204081632653, 0.007551265306122449, 0.007755326530612246, 0.00795938775510204, 0.008163448979591837, 0.008367510204081632, 0.008571571428571428, 0.008775632653061225, 0.00897969387755102, 0.009183755102040817, 0.009387816326530612, 0.009591877551020409, 0.009795938775510203, 0.01]
batch_list = [10, 51, 92, 133, 175, 216, 257, 298, 340, 381, 422, 463, 505, 546, 587, 628, 670, 711, 752, 793, 835, 876, 917, 958, 1000]
N1_list = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30]
N2_list = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60]

#hardware para
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
build_cost_model_list = [0,2,4,6,8]
infer_shapes_list = [True,False]
place_pruned_graph_list = [True,False]
enable_bfloat16_sendrecv_list = [True,False]
do_common_subexpression_elimination_list = [True,False]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [True,False]
global_jit_level_list = [0,1,2]


domain_vars = [{'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': N1_list},
                {'type': 'discrete_numeric', 'items': N2_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                {'type': 'discrete_numeric', 'items': build_cost_model_list},
                {'type': 'discrete', 'items': infer_shapes_list},
                {'type': 'discrete', 'items': place_pruned_graph_list},
                {'type': 'discrete', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list}
                ]

config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 70
moo_objectives = [runtime_eval, acc_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,config=config)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)