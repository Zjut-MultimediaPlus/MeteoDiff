# MeteoDiff
This is the code of MeteoDiff, a Generative AI with Physics-Driven Meteorological Factor Constraint for improving Tropical Cyclone Forecasting.

## Requirements 
* python 3.8.8
* Pytorch 1.11.0 (GPU)

## Data Preparation
First, we need to download all the data we used in MeteoDiff.
* MeteoDiff's processed datasets [part1](https://drive.google.com/file/d/1XpfByEZkZHAybXgB5p2YsR5KZhHrtVei/view?usp=drive_link) and [part2](https://drive.google.com/file/d/1aiJaUH035YOIbsS9Q1Y9GGmyKW1HiJU1/view?usp=drive_link)
* MeteoDiff's [checkpoint](https://drive.google.com/file/d/1H8RKJU_p1vFmIcP5gghg1BMB7oBJ8_zI/view?usp=drive_link)

After completing the downloading, move these file to correct file path.
* Move MeteoDiff's processed datasets to **/MeteoDiff**, change the **data_dir** in **MeteoDiff/configs/baseline.yaml** to **MeteoDiff/process_data_1D-ENV-ERA5-ALL-WE-NEED-vocen-vodis**
* Move MeteoDiff's checkpoint to **/MeteoDiff/experiment/MeteoDiff_ori**

## Train
```python
## change the eval_mode in MeteoDiff/configs/baseline.yaml to False ##
cd MeteoDiff
python main.py
```

## Test
```python
## change the eval_mode in MeteoDiff/configs/baseline.yaml to True ##
cd MeteoDiff
python main.py
```
