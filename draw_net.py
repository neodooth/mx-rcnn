from rcnn.symbol_resnet101 import *
from rcnn.symbol import *

for net in ['resnet']: #['vgg', 'resnet']:
    # for n in ['rpn', 'rpn_test', 'rcnn', 'rcnn_test', 'test', 'joint']:
    for n in ['joint']:
        print 'drawing', net, n
        sym = eval('get_' + net + '_' + n)()
        shape = {
            'data': (1, 3, 600, 600),
            # 'label': (1, ),
            # 'bbox_target': (1, 4 * 201),
            # 'bbox_inside_weight': (1, 4 * 201),
            # 'bbox_outside_weight': (1, 4 * 201)
        }
        net_img = mx.visualization.plot_network(sym, shape=shape, node_attrs={"fixedsize":"fasle"})
        net_img.render('net_image/' + net + '/' + n)
