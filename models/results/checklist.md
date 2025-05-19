# Results checklist

| DeepLabv3+         | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)               |
|--------------------|--------|--------|---------|------------|-----------|-----------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_04_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_04_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       | ✓          | ✓         | `c_deeplab_05_05_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_07_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_deeplab_05_07_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_05_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_deeplab_05_05_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       | ✓          | ✓         | `c_deeplab_05_08_50_cmap_adamw_cos`     |

| U-Net              | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)            |
|--------------------|--------|--------|---------|------------|-----------|--------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_06_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_06_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       | ✓          | ✓         | `c_unet_05_06_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_07_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_unet_05_07_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_07_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_unet_05_07_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       | ✓          | ✓         | `c_unet_05_08_50_cmap_adamw_cos`     |

| ViT                | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)           |
|--------------------|--------|--------|---------|------------|-----------|-------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_06_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_06_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       | ✓          | ✓         | `c_vit_05_07_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_07_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_vit_05_08_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_05_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_vit_05_05_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       | ✓          | ✓         | `c_vit_05_08_50_cmap_adamw_cos`     |

| Swin               | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)            |
|--------------------|--------|--------|---------|------------|-----------|--------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_05_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_05_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       | ✓          | ✓         | `c_swin_05_05_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_08_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_swin_05_08_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_06_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       | ✓          | ✓         | `c_swin_05_06_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       | ✓          | ✓         | `c_swin_05_08_50_cmap_adamw_cos`     |

Legend: ✓ = Saved, ✗ = Not needed/is covered

Saved in `thesis/figures/img/results`
