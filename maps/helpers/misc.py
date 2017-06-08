import os
import shutil

def make_or_clear_directory(path):
    if os.path.isdir(path):
        print 'clearing directory - ', path
        shutil.rmtree(path)
    print 'making directory - ', path
    os.makedirs(path)


if __name__ == '__main__':
    make_or_clear_directory('./test')
