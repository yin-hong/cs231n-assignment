# -*-coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
from cs231n.image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
import torchvision.transforms as T
import PIL
from torch.autograd import Variable
from scipy.misc import imread
import torch.nn as nn
import torchvision
def preprocess(img, size=512):
    transform = T.Compose([
        T.Scale(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def features_from_img(imgpath, imgsize):
    img = preprocess(PIL.Image.open(imgpath), size=imgsize)
    img_var = Variable(img.type(dtype))
    return extract_features(img_var, cnn), img_var

"""
# Older versions of scipy.misc.imresize yield different results
# from newer versions, so we check to make sure scipy is up to date.
def check_scipy():
    import scipy
    vnum = int(scipy.__version__.split('.')[1])
    assert vnum >= 16, "You must install SciPy >= 0.16.0 to complete this notebook."

check_scipy()
"""
answers = np.load('/home/hongyin/file/cs231n-assignment3/style-transfer-checks.npz')

dtype = torch.FloatTensor
# Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
# dtype = torch.cuda.FloatTensor

# Load the pre-trained SqueezeNet model.
cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(dtype)

# We don't want to train the model any further, so we don't want PyTorch to waste computation
# computing gradients on parameters we're never going to update.
for param in cnn.parameters():
    param.requires_grad = False

# We provide this helper code which takes an image, a model (cnn), and returns a list of
# feature maps, one per layer.
def extract_features(x, cnn):
    """
    Use the CNN to extract features from the input image x.

    Inputs:
    - x: A PyTorch Variable of shape (N, C, H, W) holding a minibatch of images that
      will be fed to the CNN.
    - cnn: A PyTorch model that we will use to extract features.

    Returns:
    - features: A list of feature for the input images x extracted using the cnn model.
      features[i] is a PyTorch Variable of shape (N, C_i, H_i, W_i); recall that features
      from different layers of the network may have different numbers of channels (C_i) and
      spatial dimensions (H_i, W_i).
    """
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.

    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_target: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    #pass
    scalar_content_loss = 0
    change_content_current = content_current.squeeze()
    change_content_original = content_original.squeeze()
    C_l, H_l, W_l = change_content_current.size()
    change_content_current = change_content_current.view(C_l, -1)
    change_content_original = change_content_original.view(C_l, -1)
    scalar_content_loss = content_weight * torch.sum(torch.pow((change_content_current - change_content_original), 2))
    return scalar_content_loss


def content_loss_test(correct):
    content_image = '/home/hongyin/file/cs231n-assignment3/styles/tubingen.jpg'
    image_size = 192
    content_layer = 3
    content_weight = 6e-2

    c_feats, content_img_var = features_from_img(content_image, image_size)

    bad_img = Variable(torch.zeros(*content_img_var.data.size()))
    feats = extract_features(bad_img, cnn)

    student_output = content_loss(content_weight, c_feats[content_layer], feats[content_layer]).data.numpy()
    error = rel_error(correct, student_output)
    print('Maximum error is {:.3f}'.format(error))

"""
content_loss_test(answers['cl_out'])
"""
"""
注意，此处是用gram_matrix表示图片的style特征（也可以表示texture特征?)
"""
def gram_matrix(features, normalize = True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: PyTorch Variable of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)

    Returns:
    - gram: PyTorch Variable of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    #pass
    N, C, H, W = features.size()
    reshape_features = features.view(N, C, -1)
    gram = torch.zeros(N, C, C)
    gram = Variable(gram)
    """
    for i in range(C):
        for j in range(C):
            gram[:, i, j] = torch.sum(reshape_features[:, i, :] * reshape_features[:, j, :])
    """
    reshape_features_T = torch.transpose(reshape_features, 1, 2)
    gram = torch.matmul(reshape_features, reshape_features_T)
    # print(gram)
    if normalize == True:
        gram = gram / float(H * W * C)  #  注意，这里要使用float对H * W * C进行类型转化(因为分母也是float类型 ，不然得到的结果不准确
    return gram


def gram_matrix_test(correct):
    style_image = '/home/hongyin/file/cs231n-assignment3/styles/starry_night.jpg'
    style_size = 192
    feats, _ = features_from_img(style_image, style_size)
    student_output = gram_matrix(feats[5].clone()).data.numpy()
    error = rel_error(correct, student_output)
    print('Maximum error is {:.3f}'.format(error))

"""
gram_matrix_test(answers['gm_out'])
"""

# Now put it together in the style_loss function...
def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.

    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Variable giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].

    Returns:
    - style_loss: A PyTorch Variable holding a scalar giving the style loss.
    """
    # Hint: you can do this with one for loop over the style layers, and should
    # not be very much code (~5 lines). You will need to use your gram_matrix function.
    # pass
    loss = Variable(torch.zeros(1))
    for i in range(len(style_layers)):
        cur_gram_matrix = gram_matrix(feats[style_layers[i]])
        loss_tmp = style_weights[i] * torch.sum(torch.pow((cur_gram_matrix - style_targets[i]), 2))
        loss = loss + loss_tmp

    return loss


def style_loss_test(correct):
    content_image = '/home/hongyin/file/cs231n-assignment3/styles/tubingen.jpg'
    style_image = '/home/hongyin/file/cs231n-assignment3/styles/starry_night.jpg'
    image_size = 192
    style_size = 192
    style_layers = [1, 4, 6, 7]
    style_weights = [300000, 1000, 15, 3]

    c_feats, _ = features_from_img(content_image, image_size)
    feats, _ = features_from_img(style_image, style_size)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    student_output = style_loss(c_feats, style_layers, style_targets, style_weights).data.numpy()
    error = rel_error(correct, student_output)
    print('Error is {:.3f}'.format(error))

"""
style_loss_test(answers['sl_out'])
"""

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    # Your implementation should be vectorized and not require any loops!
    #pass
    loss = Variable(torch.zeros(1))
    N, C, H, W = img.size()
    """
    for c in range(C):
        for i in range(H-1):
            for j in range(W-1):
                loss = loss + torch.sum(torch.pow((img[:, c, i, j+1] - img[:, c, i, j]), 2) + torch.pow((img[:, c, i+1, j] - img[:, c, i, j]), 2))
    """
    img_hori = torch.zeros(N, C, H, W)
    img_hori = Variable(img_hori)
    img_hori[:, :, :, :W-1] = img[:, :, :, 1:]
    img_hori[:, :, :, W-1] = img[:, :, :, W-1]
    img_ver = torch.zeros(N, C, H, W)
    img_ver = Variable(img_ver)
    img_ver[:, :, :H-1, :] = img[:, :, 1:, :]
    img_ver[:, :, H-1, :] = img[:, :, H-1, :]
    loss = torch.sum(torch.pow((img_hori - img), 2) + torch.pow((img_ver - img), 2))
    loss = tv_weight * loss

    return loss


def tv_loss_test(correct):
    content_image = '/home/hongyin/file/cs231n-assignment3/styles/tubingen.jpg'
    image_size = 192
    tv_weight = 2e-2

    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))

    student_output = tv_loss(content_img_var, tv_weight).data.numpy()
    error = rel_error(correct, student_output)
    print('Error is {:.3f}'.format(error))

"""
tv_loss_test(answers['tv_out'])
"""

def style_transfer(content_image, style_image, image_size, style_size, content_layer, content_weight,
                   style_layers, style_weights, tv_weight, init_random = False):
    """
    Run style transfer!

    Inputs:
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size= image_size)
    content_img_var = Variable(content_img.type(dtype))
    feats = extract_features(content_img_var, cnn)
    target_content = feats[content_layer]

    # Extract feature from the style image(使用gram矩阵表示style特征或者说是图像的texture)
    style_img = preprocess(PIL.Image.open(style_image), size= style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    target_styles = []
    for idx in style_layers:
        target_styles.append(gram_matrix(features=feats[idx]))


    # Initialize output image to content image or nois
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_()

    else:
        img = content_img.clone().type(dtype)

    img_var = Variable(img, requires_grad= True)

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decayed_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img_var Torch variable, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam(params=[img_var], lr= initial_lr)

    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu()))
    axarr[1].imshow(deprocess(style_img.cpu()))
    plt.show()
    plt.savefig('/home/hongyin/file/cs231n-assignment3/featureInversion/sourceImage.png')

    Iteration_num = 200

    for t in range(Iteration_num):
        if t < 190:
            img_var.clamp(-1.5, 1.5)
        optimizer.zero_grad()
        feats = extract_features(img_var, cnn)

        cont_loss = content_loss(content_weight, feats[content_layer], target_content)
        styl_loss = style_loss(feats, style_layers, target_styles, style_weights)
        t_loss = tv_loss(img_var, tv_weight)

        loss = cont_loss + styl_loss + t_loss
        loss.backward()


        if t == decayed_lr_at:
            optimizer = torch.optim.Adam([img_var], lr= initial_lr * decayed_lr)

        optimizer.step()

        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.axis('off')
            plt.imshow(deprocess(img.cpu()))
            plt.savefig('/home/hongyin/file/cs231n-assignment3/featureInversion/'+str(t)+'.png')


    print('Iteration {}'.format(t))
    plt.axis('off')
    plt.imshow(deprocess(img.cpu()))
    plt.savefig('/home/hongyin/file/cs231n-assignment3/featureInversion/final.png')





"""
# Composition VII + Tubingen
params1 = {
    'content_image' : '/home/hongyin/file/cs231n-assignment3/styles/tubingen.jpg',
    'style_image' : '/home/hongyin/file/cs231n-assignment3/styles/composition_vii.jpg',
    'image_size' : 192,
    'style_size' : 512,
    'content_layer' : 3,
    'content_weight' : 5e-2,
    'style_layers' : (1, 4, 6, 7),
    'style_weights' : (20000, 500, 12, 1),
    'tv_weight' : 5e-2
}

style_transfer(**params1)
"""

# Feature Inversion -- Starry Night + Tubingen
params_inv = {
    'content_image' : '/home/hongyin/file/cs231n-assignment3/styles/tubingen.jpg',
    'style_image' : '/home/hongyin/file/cs231n-assignment3/styles/starry_night.jpg',
    'image_size' : 192,
    'style_size' : 192,
    'content_layer' : 3,
    'content_weight' : 6e-2,
    'style_layers' : [1, 4, 6, 7],
    'style_weights' : [0, 0, 0, 0], # we discard any contributions from style to the loss
    'tv_weight' : 2e-2,
    'init_random': True # we want to initialize our image to be random
}

style_transfer(**params_inv)




