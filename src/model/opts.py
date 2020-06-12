import argparse


def configure_args():
    parser = argparse.ArgumentParser(description='Skin Condition Tracking Model')

    parser.add_argument('--model_dir', default='src',
                        help="Main directory")

    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--frame_size', type=int, default=30, help='Maximum number of frames per user')
    parser.add_argument('--image_size', type=int, default=160, help='Image width and height')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number')
    parser.add_argument('--data_size', type=int, default=10, help='Dataset size')

    parser.add_argument('--start', type=int, default=0, help='Dataset size')
    parser.add_argument('--end', type=int, default=0, help='Dataset size')

    return parser.parse_args()