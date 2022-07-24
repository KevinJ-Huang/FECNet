import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    imgs = sorted(os.listdir(inputdir))
    for idx,img in enumerate(imgs):
        groups = ''

        groups += os.path.join(inputdir, img) + '|'
        # groups += os.path.join(args.auxi, img) + '|'
        groups += os.path.join(targetdir,img)

        # if idx >= 800:
        #     break

        with open(os.path.join(outputdir, 'groups_test_mixexposure.txt'), 'a') as f:
            f.write(groups + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/jieh/Dataset/Continous/Exposure/test/input/Exp5/', metavar='PATH', help='root dir to save low resolution images')
    # parser.add_argument('--auxi', type=str, default='/home/jieh/Dataset/Continous/ExpFiveLarge/train/Mid', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--target', type=str, default='/home/jieh/Dataset/Continous/Exposure/test/target/', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/home/jieh/Projects/ExposureFrequency/FastFourierExp1/data/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()
