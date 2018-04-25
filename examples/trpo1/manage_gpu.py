import tensorflow as tf
import time


def main():
    gpu_options = tf.GPUOptions(allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    c = tf.constant(0.0, dtype=tf.float32)
    while True:
        sess.run(c)
        time.sleep(10)


if __name__ == "__main__":
    main()
