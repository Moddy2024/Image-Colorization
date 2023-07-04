import sys
sys.path.append('../')
from pycore.tikzeng import *

# Define your architecture
arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 3, 64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2),
    to_LeakyReLU("relu1", offset="(0,0,0)", to="(conv1-east)"),
    to_Conv("conv2", 64, 128, offset="(2,0,0)", to="(relu1-east)", height=35, depth=35, width=3),
    to_BatchNorm("bn2", offset="(0,0,0)", to="(conv2-east)"),
    to_LeakyReLU("relu2", offset="(0,0,0)", to="(bn2-east)"),
    to_Conv("conv3", 128, 256, offset="(2,0,0)", to="(relu2-east)", height=30, depth=30, width=4),
    to_BatchNorm("bn3", offset="(0,0,0)", to="(conv3-east)"),
    to_LeakyReLU("relu3", offset="(0,0,0)", to="(bn3-east)"),
    to_Conv("conv4", 256, 512, offset="(1.8,0,0)", to="(relu3-east)", height=23, depth=23, width=7),
    to_BatchNorm("bn4", offset="(0,0,0)", to="(conv4-east)"),
    to_LeakyReLU("relu4", offset="(0,0,0)", to="(bn4-east)"),
    to_Conv("conv5", 512, 1, offset="(1.5,0,0)", to="(relu4-east)", height=15, depth=15, width=7),
    to_connection("relu1", "conv2"),
    to_connection("conv2", "bn2"),
    to_connection("bn2", "relu2"),
    to_connection("relu2", "conv3"),
    to_connection("conv3", "bn3"),
    to_connection("bn3", "relu3"),
    to_connection("relu3", "conv4"),
    to_connection("conv4", "bn4"),
    to_connection("bn4", "relu4"),
    to_connection("relu4", "conv5"),
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
