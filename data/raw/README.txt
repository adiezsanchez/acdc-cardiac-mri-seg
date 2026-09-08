Place the official ACDC training/testing folders here after you register and
download them. This repository never ships clinical MRI.

Expected layout (matches the ACDC challenge dump):

  data/raw/training/patient001/patient001_frame01.nii.gz
  data/raw/training/patient001/patient001_frame01_gt.nii.gz
  data/raw/training/patient001/Info.cfg
  ...

Download: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

Then point configs at this directory, e.g.

  pixi run train-phase1
  # or: python -m acdc_seg train --phase 1 --data-root data/raw
