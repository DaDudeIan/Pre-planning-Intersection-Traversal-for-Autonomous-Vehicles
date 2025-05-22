# Results checklist

| DeepLabv3+         | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)               |
|--------------------|--------|--------|---------|------------|-----------|-----------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_04_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_04_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       |            |           | `c_deeplab_05_05_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_07_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_deeplab_05_07_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_05_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_deeplab_05_05_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_deeplab_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       |            |           | `c_deeplab_05_08_50_cmap_adamw_cos`     |

| U-Net              | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)            |
|--------------------|--------|--------|---------|------------|-----------|--------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_06_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_06_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       |            |           | `c_unet_05_06_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_07_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_unet_05_07_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_07_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_unet_05_07_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_unet_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       |            |           | `c_unet_05_08_50_cmap_adamw_cos`     |

| ViT                | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)           |
|--------------------|--------|--------|---------|------------|-----------|-------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_06_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_06_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       |            |           | `c_vit_05_07_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_07_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_vit_05_08_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_05_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_vit_05_05_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_vit_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       |            |           | `c_vit_05_08_50_cmap_adamw_cos`     |

| Swin               | Test 1 | Test 2 | Train 1 | Loss graph | Val graph | Checkpoint  (`file_name`)            |
|--------------------|--------|--------|---------|------------|-----------|--------------------------------------|
| CE, epoch 10       | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_05_10_ce_adamw_cos`       |
| CE, epoch 100      | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_05_100_ce_adamw_cos`      |
| CE, epoch 300      | ✓      | ✓      | ✓       |            |           | `c_swin_05_05_300_ce_adamw_cos`      |
| CE+CMap, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_08_50_ce-cmap_adamw_cos`  |
| CE+CMap, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_swin_05_08_100_ce-cmap_adamw_cos` |
| CE+Cont, Epoch 50  | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_06_50_ce-topo_adamw_cos`  |
| CE+Cont, Epoch 100 | ✓      | ✓      | ✓       |            |           | `c_swin_05_06_100_ce-topo_adamw_cos` |
| CMap, Epoch 10     | ✓      | ✓      | ✓       | ✗          | ✗         | `c_swin_05_08_10_cmap_adamw_cos`     |
| CMap, Epoch 50     | ✓      | ✓      | ✓       |            |           | `c_swin_05_08_50_cmap_adamw_cos`     |

Legend: ✓ = Saved, ✗ = Not needed/is covered

Saved in `thesis/figures/img/results`

## mIoU

| DeepLabv3+         | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | mIoU   |
|--------------------|---------|---------|---------|---------|---------|--------|
| CE, epoch 10       | 0.9695  | 0.3344  | 0.3253  | 0.2983  | 0.1772  | 0.4209 |
| CE, epoch 100      | 0.9765  | 0.3506  | 0.3188  | 0.3432  | 0.2214  | 0.4421 |
| CE, epoch 300      | 0.9774  | 0.3540  | 0.3302  | 0.3388  | 0.2480  | 0.4497 |
| CE+CMap, Epoch 50  | 0.9701  | 0.3214  | 0.3078  | 0.3216  | 0.2143  | 0.4271 |
| CE+CMap, Epoch 100 | 0.9637  | 0.2919  | 0.2532  | 0.3107  | 0.2173  | 0.4074 |
| CE+Cont, Epoch 50  | 0.9735  | 0.3053  | 0.3171  | 0.3009  | 0.2216  | 0.4237 |
| CE+Cont, Epoch 100 | 0.9711  | 0.2787  | 0.3008  | 0.2922  | 0.1846  | 0.4055 |

| U-Net              | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | mIoU   |
|--------------------|---------|---------|---------|---------|---------|--------|
| CE, epoch 10       | 0.9704  | 0.2952  | 0.2858  | 0.3276  | 0.1735  | 0.4105 |
| CE, epoch 100      | 0.9748  | 0.3016  | 0.2799  | 0.2983  | 0.1721  | 0.4053 |
| CE, epoch 300      | 0.9748  | 0.2663  | 0.2555  | 0.2692  | 0.1230  | 0.3778 |
| CE+CMap, Epoch 50  | 0.9728  | 0.3327  | 0.2892  | 0.3156  | 0.1875  | 0.4196 |
| CE+CMap, Epoch 100 | 0.9650  | 0.2984  | 0.2593  | 0.2663  | 0.1762  | 0.3930 |
| CE+Cont, Epoch 50  | 0.9722  | 0.2815  | 0.2744  | 0.3003  | 0.1353  | 0.3928 |
| CE+Cont, Epoch 100 | 0.9679  | 0.2719  | 0.2567  | 0.2841  | 0.1364  | 0.3834 |

| ViT                | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | mIoU   |
|--------------------|---------|---------|---------|---------|---------|--------|
| CE, epoch 10       | 0.9182  | 0.1472  | 0.1204  | 0.1476  | 0.0509  | 0.2769 |
| CE, epoch 100      | 0.9453  | 0.1664  | 0.1523  | 0.2065  | 0.1211  | 0.3183 |
| CE, epoch 300      | 0.9461  | 0.1549  | 0.1483  | 0.1982  | 0.1116  | 0.3118 |
| CE+CMap, Epoch 50  | 0.9218  | 0.1472  | 0.1361  | 0.2001  | 0.0890  | 0.2988 |
| CE+CMap, Epoch 100 | 0.9185  | 0.1442  | 0.1502  | 0.2162  | 0.1016  | 0.3061 |
| CE+Cont, Epoch 50  | 0.9295  | 0.1553  | 0.1593  | 0.1775  | 0.1015  | 0.3046 |
| CE+Cont, Epoch 100 | 0.9281  | 0.1617  | 0.1592  | 0.1805  | 0.1268  | 0.3113 |

| Swin               | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 | mIoU   |
|--------------------|---------|---------|---------|---------|---------|--------|
| CE, epoch 10       | 0.9348  | 0.1556  | 0.1663  | 0.1494  | 0.1417  | 0.3095 |
| CE, epoch 100      | 0.9490  | 0.2009  | 0.1895  | 0.1681  | 0.1294  | 0.3274 |
| CE, epoch 300      | 0.9485  | 0.2135  | 0.2065  | 0.1953  | 0.1271  | 0.3382 |
| CE+CMap, Epoch 50  | 0.9323  | 0.1824  | 0.1533  | 0.1579  | 0.1348  | 0.3122 |
| CE+CMap, Epoch 100 | 0.9194  | 0.1757  | 0.1413  | 0.1796  | 0.1386  | 0.3109 |
| CE+Cont, Epoch 50  | 0.9428  | 0.1691  | 0.1694  | 0.1785  | 0.1435  | 0.3207 |
| CE+Cont, Epoch 100 | 0.9385  | 0.1807  | 0.1929  | 0.1767  | 0.1340  | 0.3246 |
