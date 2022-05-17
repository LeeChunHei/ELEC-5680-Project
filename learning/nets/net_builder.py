import learning.nets.fc_2layers_1024units as fc_2layers_1024units
import learning.nets.fc_2layers_gated_1024units as fc_2layers_gated_1024units
import learning.nets.fc_2layers_512_512 as fc_2layers_512_512
import learning.nets.fc_2layers_2048_1024 as fc_2layers_2048_1024
import learning.nets.fc_2layers_256_256_256 as fc_2layers_256_256_256

def build_net(net_name, input_tfs, reuse=False):
    net = None

    if (net_name == fc_2layers_1024units.NAME):
        net = fc_2layers_1024units.build_net(input_tfs, reuse)
    elif (net_name == fc_2layers_gated_1024units.NAME):
        net = fc_2layers_gated_1024units.build_net(input_tfs, reuse)
    elif (net_name == fc_2layers_512_512.NAME):
        net = fc_2layers_512_512.build_net(input_tfs, reuse)
    elif (net_name == fc_2layers_2048_1024.NAME):
        net = fc_2layers_2048_1024.build_net(input_tfs, reuse)
    elif (net_name == fc_2layers_256_256_256.NAME):
        net = fc_2layers_256_256_256.build_net(input_tfs, reuse)
    else:
        assert False, 'Unsupported net: ' + net_name
    
    return net