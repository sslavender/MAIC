{
    "name": "MAIC_CelebA",
    "mode": "sr_align",
    "gpu_ids": [0, 1],
    "use_tb_logger": true,
    "scale": 8,
    "is_train": true,
    "rgb_range": 1,
    "save_image": true,
    "datasets": {
        "train": {
            "mode": "HRLandmark",
            "name": "CelebALandmarkTrain",
            "dataroot_HR": "datasets/img_celeba",
            "info_path": "datasets/CelebA_train.pkl", 
            "data_type": "img",
            "n_workers": 8,
            "batch_size": 6,
            "LR_size": 16,
            "HR_size": 128,
            "use_flip": false,
            "use_rot": false,
            "sigma": 1
        },
        "val": {
            "mode": "HRLandmark",
            "name": "CelebALandmarkVal",
            "dataroot_HR": "datasets/img_celeba",
            "info_path": "datasets/CelebA_val.pkl",
            "data_type": "img",
            "LR_size": 16,
            "HR_size": 128,
            "sigma": 1
        }
    },
    "networks": {
        "which_model": "MAIC",
        "num_features": 330,
        "in_channels": 3,
        "out_channels": 3,
        "num_steps": 4,
        "num_groups": 6,
        "detach_attention": false,
        "hg_num_feature": 256,
        "hg_num_keypoints": 68,
        "num_fusion_block": 7
    },
    "solver": {
        "type": "ADAM",
        "learning_rate": 1e-4,
        "weight_decay": 0,
        "lr_scheme": "MultiStepLR",
        "lr_steps": [
            1e4, 2e4, 4e4, 8e4
        ],
        "lr_gamma": 0.5,
        "manual_seed": 0,
        "save_freq": 5e3,
        "val_freq": 2e3,
        "niter": 1.5e5,
        "num_save_image": 20,
        "log_full_step": true,
        "pretrain": false,
        "release_HG_grad_step": 2e3,
        "HG_pretrained_path": "models/FB_HG_68_CelebA.pth",
        "loss": {
            "pixel": {
                "loss_type": "l1",
                "weight": 1
            },
            "align": {
                "loss_type": "l2",
                "weight": 1e-1
            }
        }
    },
    "logger": {
        "print_freq": 250
    },
    "path": {
      "root": "../"
    }
}

