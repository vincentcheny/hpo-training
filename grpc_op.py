from __future__ import absolute_import, division, print_function
import tensorflow as tf
import grpc
import transfer_pb2
import transfer_pb2_grpc
import pickle


def push(trainable_var):
    channel = grpc.insecure_channel('localhost:20001')
    stub = transfer_pb2_grpc.TransferStub(channel)
    para_list = []
    for var in trainable_var:
        para_list.append(var.numpy())
    para_list = pickle.dumps(para_list)
    response = stub.UploadPara(transfer_pb2.UploadRequest(para=para_list))
    # print("push response:", response.message)


def clear(trainable_var):
    for var in trainable_var:
        var.assign(tf.zeros(shape=var.shape, dtype=var.dtype))
    # print("After grpc clear, trainable_var becomes:", trainable_var)


def pull(trainable_var):
    channel = grpc.insecure_channel('localhost:20001')
    stub = transfer_pb2_grpc.TransferStub(channel)
    response = stub.DownloadPara(transfer_pb2.DownloadRequest())
    if response is not None:
        response = pickle.loads(response.message)
        # print("pull response:", response)
        for i in range(len(trainable_var)):
            tmp = response[i]
            # print(tmp)
            trainable_var[i].assign(tf.convert_to_tensor(tmp, dtype=trainable_var[i].dtype))