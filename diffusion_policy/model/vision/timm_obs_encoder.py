import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
import numpy as np

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.tactile.mae import MAE_ViT, MAE_Encoder

logger = logging.getLogger(__name__)

def global_pool_nlc(
        x: torch.Tensor,
        pool_type: str = 'token',
        num_prefix_tokens: int = 1,
        reduce_include_prefix: bool = False,
):
    if not pool_type:
        return x

    if pool_type == 'token':
        x = x[:, 0]  # class token
    else:
        x = x if reduce_include_prefix else x[:, num_prefix_tokens:]
        if pool_type == 'avg':
            x = x.mean(dim=1)
        elif pool_type == 'avgmax':
            x = 0.5 * (x.amax(dim=1) + x.mean(dim=1))
        elif pool_type == 'max':
            x = x.amax(dim=1)
        else:
            assert not pool_type, f'Unknown pool type {pool_type}'

    return x

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class MAEloss(nn.Module):
    def __init__(self, feature_dim=768, action_dim=20, latent_dim=128):
        super().__init__()
        self.image_projection = nn.Linear(feature_dim*2 + latent_dim, feature_dim)
        self.tactile_projection = nn.Linear(feature_dim*2 + latent_dim, feature_dim)
        self.proprio_projection = nn.Linear(action_dim, latent_dim)

    def forward(self, image_features, text_features, obs_dict):
        proprio = torch.cat([obs_dict['robot0_eef_pos'], obs_dict['robot0_eef_rot_axis_angle'], obs_dict['robot0_gripper_width']], dim=-1)
        proprio = proprio.reshape(proprio.shape[0], -1)
        proprio = self.proprio_projection(proprio) # b, 128

        past_multimodal_features = torch.cat([image_features[:, 0, :], text_features[:, 0, :], proprio], dim=-1) # b, 768*2
        predicted_tac_feature = self.tactile_projection(past_multimodal_features) # b, 768
        predicted_img_feature = self.image_projection(past_multimodal_features) # b, 768

        total_loss = (F.mse_loss(predicted_tac_feature, text_features[:, -1, :]) +
                      F.mse_loss(predicted_img_feature, image_features[:, -1, :])) / 2
        return total_loss
class ClipLoss(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.tactile_projection = nn.Linear(feature_dim, 512)
        self.image_projection = nn.Linear(feature_dim, 512)

    def get_logits(self, image_features, text_features, logit_scale):
        image_features = self.image_projection(image_features)
        text_features = self.tactile_projection(text_features)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logit_scale = self.logit_scale.exp()
        logit_scale = torch.clamp(logit_scale, min=1e-6, max=1000)
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
        total_loss = (
                     F.cross_entropy(logits_per_image, labels) +
                     F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            repr_loss: str='clip',
            image_size: int=224,
            feature_dim: int=768,
            low_res: bool=False,
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_shape_map = dict()
        self.low_res = low_res
        print('low_res:', low_res)

        assert global_pool == ''

        if model_name.startswith('vit'):
            self.image_model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,  # '' means no pooling
                num_classes=0  # remove classification layer
            )  # use pretrained image vit
            self.tactile_model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool,  # '' means no pooling
                num_classes=0  # remove classification layer
            )  # use pretrained image vit
        else:
            raise NotImplementedError

        if frozen:
            assert pretrained
            for param in self.image_model.parameters():
                param.requires_grad = False
            for param in self.tactile_model.parameters():
                param.requires_grad = False

        if use_group_norm and not pretrained:
            self.image_model = replace_submodules(
                root_module=self.image_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )
            self.tactile_model = replace_submodules(
                root_module=self.tactile_model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )

        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        # data_config = timm.data.resolve_model_data_config(self.image_model)
        # model_eval_transform = timm.data.create_transform(**data_config, is_training=False)
        # normalization = model_eval_transform.transforms[-1]
        # print('eval trasform', model_eval_transform)
        # print('normalization:', normalization)
        # train_resize_crop = [torchvision.transforms.RandomResizedCrop(
        #     size=(image_size, image_size),
        #     scale=(0.85, 1.0),
        #     ratio=(0.9, 1.1),
        #     antialias=True
        # )]
        #
        # eval_resize_crop = [
        #     torchvision.transforms.Resize(
        #         size=(int(image_size//0.95), int(image_size//0.95)),
        #         antialias=True
        #     ),
        #     torchvision.transforms.CenterCrop(size=(image_size, image_size))
        # ]

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                             torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                             torchvision.transforms.Resize(size=image_shape[0], antialias=True)
                         ] + transforms[1:]
            if imagenet_norm:
                transforms = transforms + [
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        train_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        if self.low_res:
            self.low_res_transform = nn.Sequential(
                torchvision.transforms.Resize(size=(16, 16), antialias=True), # resize to low res
                torchvision.transforms.Resize(size=image_shape[0], antialias=True) # resize to original size
            )
        else:
            self.low_res_transform = nn.Identity()

        eval_transforms = None
        if transforms is not None:
            eval_transforms = [torchvision.transforms.Resize(size=image_shape[0], antialias=True)]
            if imagenet_norm:
                eval_transforms = eval_transforms + [
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        eval_transform = nn.Identity() if transforms is None else torch.nn.Sequential(*eval_transforms)

        print('new train transforms:', train_transform)
        print('new eval transforms:', eval_transform)

        self.image_train_transform = train_transform
        self.image_eval_transform = eval_transform

        if repr_loss == 'clip':
            self.repr_loss = ClipLoss(feature_dim=feature_dim)
            self.repr_loss_name = 'clip'
            print('using clip loss')
        elif repr_loss == 'mae':
            obs_shape_meta = shape_meta['obs']
            horizon = obs_shape_meta['robot0_eef_pos']['horizon']
            total_dim = 3 + 6 + 1
            self.repr_loss = MAEloss(feature_dim=feature_dim, action_dim=horizon*total_dim)
            self.repr_loss_name = 'mae'
            print('using mae loss')
        elif repr_loss == 'none':
            self.repr_loss = None
            self.repr_loss_name = 'none'
            print('using no repr loss')
        else:
            raise NotImplementedError

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if 'clip' in self.model_name:
            return feature[:, 0, :] # cls token for clip
        else:
            num_prefix_tokens = self.image_model.num_prefix_tokens
            feature = global_pool_nlc(feature, pool_type='max', num_prefix_tokens=num_prefix_tokens)
            return feature

    def forward(self, obs_dict):
        features = list()
        repr_features = dict()
        batch_size = next(iter(obs_dict.values())).shape[0]

        # process image input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T, C, H, W = img.shape
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])

            if 'right' in key: # b*t, c, h, w
                # flip right image to left image
                after_img = torch.flip(img, dims=[-1])
                img = after_img
            # add different image transformation for train and eval
            if self.training:
                img = self.image_train_transform(img)
            else:
                img = self.image_eval_transform(img)

            if 'tactile' in key:
                if self.low_res:
                    img = self.low_res_transform(img)
                raw_feature = self.tactile_model(img)
            else:
                raw_feature = self.image_model(img)

            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))
            if self.repr_loss_name == 'clip':
                repr_features[key] = feature # B*T, 768
            else:
                repr_features[key] = feature.reshape(B, T, -1)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        image_feature = repr_features['camera0_rgb']
        # with 0.5 probablity left otherwise choose right
        if np.random.rand() > 0.5:
            tactile_feature = repr_features['camera0_right_tactile']
        else:
            tactile_feature = repr_features['camera0_left_tactile']

        if self.repr_loss_name == 'clip':
            repr_loss = self.repr_loss(image_feature, tactile_feature)
        elif self.repr_loss_name == 'mae':
            repr_loss = self.repr_loss(image_feature, tactile_feature, obs_dict)
        elif self.repr_loss_name == 'none':
            repr_loss = torch.tensor(0.0, device=self.device)
        else:
            raise NotImplementedError

        repr_loss = repr_loss * 0.5
        return result, repr_loss

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            print(this_obs.shape)
            example_obs_dict[key] = this_obs
        example_output, _ = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
