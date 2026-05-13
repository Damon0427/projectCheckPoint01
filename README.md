# projectCheckPoint01

## Author
- Zicheng Tan

## File Structure

```
projectCheckPoint01/
├── DownloadScript/
│   └── datasetScript.py                    
├── MediaPipeModel/
│   └── blaze_face_short_range.tflite
├── report/   
│   └── Dual_Channel_Deepfake_Detection.pdf
├── result/   
│   └── Confusion Matrix.png
│   └── ROC.png
│   └── cross_dataset.png
│   └── Performance_Matrix.png
├── Dual_Channel_Deepfake_Detection.ipynb
├── evaluate.py  
├── models.py
├── training.py
├── preprocess.py
├── README.md
└── Requiment.txt
```
### File Description:
```
- datasetScript.py                         # A download Script From FaceForensics++.
- blaze_face_short_range.tflite            # A model from mediapipe documentation. https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector
- Requiment.txt                            # All of the require dependencies
- evaluate.py,models.py, training.py, preprocess.py  # script for train,evaluate preprocess.
- Result Folder included all of the plot. 
- Github Restriced the file size, Unable to upload the model
```

## How to set up the environment & Run：
This project was developed and tested in Google Colab.  
All project files were stored in Google Drive. 


1. Downloading the dataset 

    Run the commant below to download the original video
    - if you want to download another version(Deepfake, or FaceSwap, and so on), simplyty change "original" to "Deepfake"

    ```
    # download 150 original videos with compression level c23.
     python3 datasetScript.py ./data -d original -c c23 -t videos -n 150 --server EU2           
    ```
   - (Optional) Cross Dataset Evaluation:
         Celeb Dataset: https://github.com/yuezunli/celeb-deepfakeforensics.
         Once you filed out the google form, the download link will be provided. 

2. Open `Project_CheckPoint1.ipynb` in Google Colab.
3. Mount your Google Drive in Colab:
4. Install the required packages:
    ```
    !pip install -r Requiment.txt
    ```
5. Make sure the dataset file, model file, and notebook are located in the correct paths. 
    ```
        In my notebook, It setup base on Google Colab, If you have all of the datset file, model store in local, then You could simplyly change the "deepfake_dir" , "original_dir", and "save_dir" to your local file address.
    ```

6. run all the cell!

## Here are the shortcuts to download all of the data file:
MediaPipe:
https://drive.google.com/drive/folders/1togDCyjosDnJqZidC0J3GvMVKMvhcGoV?usp=sharing

DataFile:
https://drive.google.com/drive/folders/1BYLALxjiilcdV0dVeVbM9rew7CIwmW_N?usp=sharing
