import argparse
import os

from model import pix2pix
import tensorflow as tf

parser = argparse.ArgumentParser(description='Image to image deep learning with GAN and Deconv')
parser.add_argument('--data_dir', default='/images', help='dir of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', type=int, default=1140, help='scale images to this size')
parser.add_argument('--fine_size', type=int, default=1024, help='then crop to this size')
parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--phase', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', type=float, default=100.0, help='weight on L1 term in objective')
parser.add_argument('--use_gan', help='keep the gan loss term', default=True, action='store_true')
parser.add_argument('--use_L1', help='keep the l1 loss term', default=True, action='store_true')
parser.add_argument('--use_wgan', help='use wgan for gan loss term', default=False, action='store_true')

args = parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    with tf.Session() as sess:
        model = pix2pix(sess, args) 

        if args.phase == 'train':
            model.train()
        else:
            model.test()

if __name__ == '__main__':
    tf.app.run()
