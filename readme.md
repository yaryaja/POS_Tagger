1.Before running the code, make sure you have installed the following libraries
	
    • conllu
    • pandas
    • torch

install the conllu by doing !pip install conllu  , check the version compatibility

2. For loading the data, give the path of the data in load_data function call for both train and test

3.Change saved=True  for doing the prediction with the pre-trained model. Else it will start the training the model again.

4.Inside condition if saved==True: , give the path of the pretrained model in line 

5. After runing the code, it will after printing the accuracy, it will ask for input sentence for prediting the tags.

6. Give an input sentece , the model will give corresponding pos tags.



## the google drive link for the pretrined model are 
  #  for ffnn pos tagger : https://drive.google.com/file/d/1-CdWGMtqfcBk-nBWJ1j71IUTbTts7F2_/view?usp=drive_link
  #  for lstm pos tagger : https://drive.google.com/file/d/1k-Wy4TH-stMYzRAS_5L58DpjNm0hSJKv/view?usp=drive_link