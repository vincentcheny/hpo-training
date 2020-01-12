## How to regenerate the chart

1. Preparation

   - Env
     - ``tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl``
     - ``tensorflow_datasets``

   - Model
     - https://github.com/vincentcheny/GRPC-Model/blob/master/worker_model_vgg/worker.py

   - Device
     - A bad USB flash disk

2. Steps

   1. Mount the USB flash disk and set ``model_dir`` on it

   2. Run the code (w0 and w1)

      ```bash
      $ python worker.py --model_dir /mnt/f/estimator --save_ckpt_steps 50 --use_original_ckpt False --task_index 0 # the n-th worker uses task_index n
      ```

   3. Restart w1 at step 80

   4. Terminate at step 120

      - We can set the ``max_steps`` at ``TrainSpec`` to terminate the program automatically.

3. Things to mention

   - The latest whl is not fully compatible with my codes about tf summary. The first time we run the model there will be a ``tensorflow.python.framework.errors_impl.DataLossError``. To solve it temporarily, just start all workers again. Or we can use the old tf1.14 whl if training speed doesn't matter.
   - When generating the chart, the USB disk we use is not slow enough. So we copy some big files to it in order to block the IO. Ckpt time before blocking: 17s, after blocking: 55s.

