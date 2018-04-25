import tensorflow as tf
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description='Script to manage gpu')
    parser.add_argument('--t', required=False, default=5 * 60, type=float,
                        help='process time')
    args = parser.parse_args()
    gpu_options = tf.GPUOptions(allow_growth=False)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
    c = tf.constant(0.0, dtype=tf.float32)
    sess.run(c)
    time.sleep(args.t)


if __name__ == "__main__":
    main()
