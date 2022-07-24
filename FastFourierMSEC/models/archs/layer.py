# based on https://github.com/rogertrullo/pytorch_convlstm
import torch.nn as nn
from torch.autograd import Variable
import torch
from functools import partial
from torch.nn.init import xavier_normal_ , kaiming_normal_


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()

        # self.shape = shape#H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        # self.batch_size=batch_size
        self.padding = (filter_size - 1) // 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        # print 'hidden ',hidden.size()
        # print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        # print 'combined',combined.size()
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size, num_layers=1):
        super(CLSTM, self).__init__()

        # self.shape = shape#H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        cell_list = []
        cell_list.append(CLSTM_cell(self.input_chans, self.filter_size, self.num_features).cuda())  # the first
        # one has a different number of input channels

        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.num_features, self.filter_size, self.num_features).cuda())
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """

        current_input = input.transpose(0, 1)  # now is seq_len,B,C,H,W
        # current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        for idlayer in range(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels
            all_output = []
            output_inner = []
            for t in range(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](current_input[t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,H,W

        return next_hidden, current_input

    def init_hidden(self, batch_size, shape):
        init_states = []  # this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, shape))
        return init_states



def get_weight_init_fn( activation_fn  ):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    if hasattr( activation_fn , 'func' ):
        fn = activation_fn.func

    if  fn == nn.LeakyReLU:
        negative_slope = 0
        if hasattr( activation_fn , 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr( activation_fn , 'args'):
            if len( activation_fn.args) > 0 :
                negative_slope = activation_fn.args[0]
        return partial(kaiming_normal_,  a = negative_slope )
    elif fn == nn.ReLU or fn == nn.PReLU :
        return partial(kaiming_normal_, a = 0 )
    else:
        return xavier_normal_
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , activation_fn= None , use_batchnorm = False , pre_activation = False , bias = True , weight_init_fn = None ):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn
    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    try:
        weight_init_fn( conv.weight )
    except:
        print( conv.weight )
    layers.append( conv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , activation_fn = None ,   use_batchnorm = False , pre_activation = False , bias= True , weight_init_fn = None ):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn
    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    deconv = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( deconv.weight )
    layers.append( deconv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def linear( in_channels , out_channels , activation_fn = None , use_batchnorm = False ,pre_activation = False , bias = True ,weight_init_fn = None):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn
    examples:
        linear(3,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    linear = nn.Linear( in_channels , out_channels )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( linear.weight )

    layers.append( linear )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """
    def __init__(self, in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU ,  inplace=True ) , last_activation_fn = partial( nn.ReLU , inplace=True ) , pre_activation = False , scaling_factor = 1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn , use_batchnorm )
        self.conv2 = conv( out_channels , out_channels , kernel_size , 1 , kernel_size//2 , None , use_batchnorm  , weight_init_fn = get_weight_init_fn(last_activation_fn) )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm )
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor
    def forward(self , x ):
        residual = x
        if self.downsample is not None:
            residual = self.downsample( residual )

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out