# San Francisco Crime Classification

The required directory structure is maintained [here](https://drive.google.com/drive/folders/1jThdeSwfb2dE0OGHMO5QgCVyY7ajRCrR?usp=sharing) in the Google Drive folder.
The pickle models are maintained in this [`pickle` folder](https://drive.google.com/drive/folders/1wySWcpjyqn2RoEHB0DJbivcMbBIWm0r8?usp=sharing) and the associated dataset (along with the cleaned data) has been maintained in this [`dataset` folder](https://drive.google.com/drive/folders/1g7Izb1jGPjNEC91VHnVZOjr57CVpchq2?usp=sharing).

__Refer to the [project report](ML_Project_report.pdf) for a detailed analysis of our project and how we proceeded with it.__

### Install Requirements
```bash
pip3 install -r requirements.txt
```

### Perform Feature Engineering
```
python3 feature_engineering.py
```

### Visualization / EDA
```
python3 visualization.py
```

### Build models (as required) and pickle the same
```
python3 model.py
```
