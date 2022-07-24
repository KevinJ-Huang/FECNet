import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    imgs = os.listdir(inputdir)
    for img in imgs:
        groups = ''

        groups += os.path.join(inputdir, img) + '|'
        groups += os.path.join(targetdir,img)

        with open(os.path.join(outputdir, 'groups_test_lowReFive.txt'), 'a') as f:
            f.write(groups + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/gdata/huangjie/Continous/ExpFive/test/Low', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/gdata/huangjie/Continous/ExpFive/test/Retouch', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/ghome/huangjie/Continous/Baseline/', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()
