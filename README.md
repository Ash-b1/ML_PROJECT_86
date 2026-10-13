# Automatic Image Colorization (SVR + MRF)

A recreation of **"Automatic Image Colorization"** by Harrison Ho and Varun Ramesh (Stanford). Given a grayscale photo of a landscape, the code predicts its colors with no user input.


## Setup

Requires Python 3.9 or newer.

```
pip install -r requirements.txt
```

### 1. Build the dataset

```
python getting_data.py --contact your@email.com
```

### 2. Train

```
python train_svr.py -t data/train/
```

### 3. Test and evaluate

**Colorize one image**

```
python test_svr.py path/to/image.jpg
```

Saves `out.png` (the prediction) and `out_compare.png` (original | grayscale | prediction). Add `-s result.png` to choose the filename and `--show` to display it.

**Evaluate on a folder of color test images**

```
python test_svr.py data/test/
```

Prints the average error for the grayscale input, the regressor alone, and the regressor plus ICM, like Figure 6 of the paper. The error is `‖predicted RGB − true RGB‖ / (height × width)`.

**Compare experiments**

```
python test_svr.py data/test/ --models-dir models_improved
python test_svr.py data/test/ --sweep         # compare all models from train_svr.py --sweep
python test_svr.py data/test/ --tune-icm      # sweep the ICM weight and threshold
```

`test_svr.py` reads `<models-dir>/config.json` and builds features exactly as training did, so we don't need to repeat the feature flags at test time.

## ERROR

### **Error before any features added**
Average error over 60 test images (gamma=2, sigma^2=0.25, threshold=25):

  Before processing (grayscale): 3.593e-04
  
  With regressor only (no ICM):  3.444e-04   (4.1% lower than grayscale)
  
  With regressor and ICM:        3.422e-04   (4.8% lower than grayscale)



### **Error after luminescence histogram and gradient features added**

Average error over 60 test images (gamma=2, sigma^2=0.25, threshold=25):

  Before processing (grayscale): 3.593e-04
  
  With regressor only (no ICM):  3.416e-04   (4.9% lower than grayscale)
  
  With regressor and ICM:        3.334e-04   (7.2% lower than grayscale)

### **Error after we convert RGB to CIELAB instead of YUV**

Average error over 57 test images (gamma=2, sigma^2=0.25, threshold=25, boost=1):

  Before processing (grayscale): 3.633e-04

  With regressor only (no ICM):  3.400e-04   (6.4% lower than grayscale)
  
  With regressor and ICM:        3.397e-04   (6.5% lower than grayscale)