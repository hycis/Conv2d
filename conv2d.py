
from pylearn2.config import yaml_parse

from pylearn2.utils import sharedX

def experiment(state, channel):

    train = """
    !obj:pylearn2.train.Train {
        dataset: &train !obj:faceEmo.FaceEmo {
            which_set: 'train',
            one_hot: 1
        },
        model: !obj:pylearn2.models.mlp.MLP {
            batch_size: 100,
            input_space: !obj:pylearn2.space.Conv2DSpace {
                shape: [48, 48],
                num_channels: 1
            },
            layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                         layer_name: 'h2',
                         output_channels: 32,
                         irange: .05,
                         kernel_shape: [7, 7],
                         pool_shape: [3, 3],
                         pool_stride: [2, 2],
                         max_kernel_norm: 1.9365
                     }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                         layer_name: 'h3',
                         output_channels: 32,
                         irange: .05,
                         kernel_shape: [7, 7],
                         pool_shape: [3, 3],
                         pool_stride: [2, 2],
                         max_kernel_norm: 1.9365
                     }, !obj:pylearn2.models.mlp.Softmax {
                         max_col_norm: 1.9365,
                         layer_name: 'y',
                         n_classes: 3,
                         istdev: .05
                     }
                    ],
        },
        algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 100,
            learning_rate: .01,
            init_momentum: .5,
            monitoring_dataset:
                {
                    'valid' : !obj:faceEmo.FaceEmo {
                                  which_set: 'test',
                                  one_hot: 1,
                              },
                    'test'  : !obj:faceEmo.FaceEmo {
                                  which_set: 'test',
                                  one_hot: 1,
                              }
                },
            cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
                !obj:pylearn2.costs.cost.MethodCost {
                    method: 'cost_from_X'
                }, !obj:pylearn2.costs.mlp.WeightDecay {
                    coeffs: [ .00005, .00005, .00005 ]
                }
                ]
            },
            termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
                channel_name: "valid_y_misclass",
                prop_decrease: 0.,
                N: 10
            }
        },
        extensions:
            [ !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                 channel_name: 'valid_y_misclass',
                 save_path: "convolutional_network_best.pkl"
            }, !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                start: 1,
                saturate: 10,
                final_momentum: .99
            }
        ]
    }
    """


 
    train = yaml_parse.load(train)
    train.algorithm.learning_rate = sharedX(state.lr, 'learning_rate')
    train.algorithm.batch_size = state.batch_size
    print 'learning rates', train.algorithm.learning_rate
    print 'batch size', train.algorithm.batch_size
    train.main_loop()
    
    return channel.COMPLETE
