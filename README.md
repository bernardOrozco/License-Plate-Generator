# License-Plate-Generator

A simple code for creating licence plate images for licence plate decoder and licenc plate detection

**Note**
for background images we downloades SUN2012pascalformat and liked the images folder

## Run

for creating just the plate with label

```sh
$ python create_train_data.py
```

for creating detection with bounding box and labels of the plate

```sh
$ python create_detect_data.py
```

## Todo

[x] Create detection data with csv
[x] Create decoding data with csv
[ ] Add argparse to variables
[ ] Create more types of plate with more templates
