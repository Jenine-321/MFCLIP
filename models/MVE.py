import torch
from torch import nn
from einops import rearrange
from skimage import io, color
import skimage
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import cv2

class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3, learnable=False):
        super(SRMConv2d_simple, self).__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)  # (3,3,5,5)
        self.kernel = nn.Parameter(data=kernel, requires_grad=learnable)
        # self.hor_kernel = self._build_kernel().transpose(0,1,3,2)

    def forward(self, x):
        '''
        x: imgs (Batch, H, W, 3)
        '''
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        out = self.truc(out)

        return out

    def _build_kernel(self, inc):
        # filter1: KB
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        # filter2：KV
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        # filter3：hor 2rd
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, -2, 1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 4.
        filter2 = np.asarray(filter2, dtype=float) / 12.
        filter3 = np.asarray(filter3, dtype=float) / 2.
        # statck the filters
        filters = [[filter1],  # , filter1, filter1],
                   [filter2],  # , filter2, filter2],
                   [filter3]]  # , filter3, filter3]]  # (3,3,5,5)
        filters = np.array(filters)
        filters = np.repeat(filters, inc, axis=1)
        filters = torch.FloatTensor(filters)  # (3,3,5,5)
        return filters


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

def Patch_Selector (img_tensor, patch_size):

    c, h, w = img_tensor.size()
    img_tensor = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    img_tensor = img_tensor.contiguous().view(c, -1, patch_size, patch_size)
    patches = img_tensor.permute(1, 0, 2, 3)

    homogeneity_scores = []
    to_grayscale = transforms.Grayscale(num_output_channels=1)

    for patch in patches:
        patch = to_grayscale(patch)
        patch = patch.squeeze()
        patch = patch.cpu().numpy() if patch.is_cuda else patch.numpy()
        glcm = skimage.feature.graycomatrix((patch * 255).astype(np.uint8), distances=[1], angles=[0], levels=256,
                                            symmetric=True,
                                            normed=True)
        # 计算同质性
        homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]
        homogeneity_scores.append(homogeneity)

    max_homogeneity_index = np.argmin( homogeneity_scores)
    rich_patch = patches[max_homogeneity_index]
    rich_patch = rich_patch.unsqueeze(0).cuda()

    return rich_patch



class MVE(nn.Module):
    def __init__(self, image_size=224, patch_size=7, richest_patch_size = 112, channels=512,
                 dim=1024, depth=6, heads=8, mlp_dim=2048):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.features_SRM= nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        num_patches = (7 // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        patch_dim_srm = 64 * (richest_patch_size//4) ** 2
        self.richest_patch_size = richest_patch_size

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(24, 1, dim))
        self.pos_embedding_srm = nn.Parameter(torch.randn(24, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.cls_token_rm = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer_srm = Transformer(dim, 3, heads, mlp_dim)
        self.patchsrm_to_embedding = nn.Linear(patch_dim_srm, dim)
        self.to_cls_token = nn.Identity()
        self.SRM = SRMConv2d_simple()

    def forward(self, img, mask=None):
        p = self.patch_size
        numpy_array = img.detach().cpu().numpy()
        rich_patches = []
        rgb_images = []
        for i in range(numpy_array.shape[0]):
            image = numpy_array[i]
            image_rgb = cv2.resize(image, (224, 224))
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
            arr_transposed = image_rgb.transpose((2, 0, 1))
            arr_expanded = np.expand_dims(arr_transposed, axis=0)
            image_tensor = torch.tensor(arr_expanded).float() / 255
            image_tensor = image_tensor.squeeze(0)
            rich_patch = Patch_Selector(image_tensor, patch_size= self.richest_patch_size)
            out = self.SRM(rich_patch).squeeze(0) # tensor 1,3,32,32
            rich_patches.append(out.detach().cpu().numpy())
            rgb_images.append(image_rgb)

        img_rich_patches = torch.tensor(rich_patches).float().cuda()
        img_rgb_t = torch.tensor(rgb_images).cuda()
        img_rgb_t = img_rgb_t.permute(0, 3, 2, 1).float()

        x_rgb = self.features(img_rgb_t)
        x_srm = self.features_SRM(img_rich_patches)
        y_srm = rearrange( x_srm, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=28, p2=28)
        y = rearrange(x_rgb, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        y = self.patch_to_embedding(y)

        y_srm = self.patchsrm_to_embedding(y_srm)

        cls_tokens = self.cls_token.expand(x_rgb.shape[0], -1, -1)
        cls_tokens_srm = self.cls_token_rm.expand(x_srm.shape[0], -1, -1)

        x_srm = torch.cat((cls_tokens_srm, y_srm), 1)
        x = torch.cat((cls_tokens, y), 1)

        shape = x.shape[0]
        x_srm += self.pos_embedding_srm[0:shape]
        x += self.pos_embedding[0:shape]
        x_srm = self.transformer_srm(x_srm, mask)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x_srm = self.to_cls_token(x_srm[:, 0])
        img_noise_fusion = x + x_srm

        return img_noise_fusion

