# DSP-Rijks-Project

To make the code run follow the steps below: 
* IN TERMINAL: git clone https://github.com/VanshitaSharmaKumar/DSP-Rijks-Project.git
* create virtual environment IN TERMINAL: python -m venv venv 
* IN TERMINAL: pip install -r requirements.txt

### Main Features
* Text feature extraction using Transformers and BERT.
* Image feature extraction using ResNet-50.
* Merging of text and image features to create personalized art recommendations.
* Streamlit integration.
* deployment on ngrok

### How to run the program step by step:
Once the repository has been cloned, we will first extract the text features using TF-IDF from the meta data. 
* IN TERMINAL: python build_rijks_files.py

A folder names DATA should be created, in DATA you should be able to see the following files:
*  final_features.npy file should be created in the DATA folder

Next, the featurs are then used in app.py. As of now, we are only using the text features to create recommendations, to run this prior to image features we do the following command
* IN TERMINAL: streamlit run app.py 

You can verify if code works by running the above. 

Follow the next steps for image features extraction, using ResNet-50 which is then merged with text features to complete the recommendation system. 
* IN TERMINAL: python image_extraction.ipynb
* IN TERMINAL: python build_final_features.py
* IN TERMINAL: streamlit run app.py 