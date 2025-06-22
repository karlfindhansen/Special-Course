# Special-Course
Using Deep Learning to predict bed topography in Greenland.

# Guide

1. Upload data to the correct folders. The data and where to find it, is outlined in the report. 

2. Run the script to find locations:
```bash
python data/find_location.py
```
Here you must specify if you want to find it for a specic glaicer, just write its name and it will generate crops to it.

3. Ensure that you can run the dataloader
```bash
python data/data_preprocessing.py
```
This should be possible, otherwise there may be issues with the data. Don't hessitate to contact me.

4. Now the model can be trained
```bash
python src/train.py
```
Progress is printed each epoch.