# GRPC-Model

## Introduction

Generate a set of trainable variable backup by GRPC for multi-worker distributed training with Keras model. 

## Usage

### Plug-in GRPC server

```shell
$ cd server_client_model
$ python transfer_server.py
$ python client_0.py
$ python client_1.py
# The parameter values in client 1 will be set to zero and it will copy the values from GRPC server backup.
```





