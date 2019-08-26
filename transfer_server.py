# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC transfer server."""

from concurrent import futures
import time
import logging

import grpc
import transfer_pb2
import transfer_pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

trainable_var_backup = None


class Transfer(transfer_pb2_grpc.TransferServicer):

    def UploadPara(self, request, context):
        global trainable_var_backup
        if trainable_var_backup is not request.para:
            trainable_var_backup = request.para
        return transfer_pb2.UploadReply(message=trainable_var_backup)

    def DownloadPara(self, request, context):
        global trainable_var_backup
        return transfer_pb2.DownloadReply(message=trainable_var_backup)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    transfer_pb2_grpc.add_TransferServicer_to_server(Transfer(), server)
    server.add_insecure_port('[::]:20001')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    logging.basicConfig()
    serve()
