import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from io import open
from conllu import parse_incr

def load_data(file_loc):
  data_file = open(file_loc, "r", encoding="utf-8")

  X_sentence=[]
  Y_sentence=[]

  for sentence in parse_incr(data_file):
      X_sentence.append([sentence.metadata['text']])
      temp=[]
      for i in range(len(sentence.metadata['text'].split())):
        temp.append(sentence[i]['upos'])
      Y_sentence.append(temp)
  return X_sentence,Y_sentence

#word_tag_pairs
def data_preparation(x,y):
  data=[]
  for sentence,tags in zip(x,y):
    sentence=sentence[0].split()
    word_tag_pairs=list(zip(sentence, tags))
    data.append(word_tag_pairs)
  return data

def calculate_index(all_data):
  word_to_idx = {}
  tag_to_idx = {}
  char_to_idx = {}
  for sentence in all_data:
      for word, pos_tag in sentence:
          if word not in word_to_idx.keys():
              word_to_idx[word] = len(word_to_idx)
          if pos_tag not in tag_to_idx.keys():
              tag_to_idx[pos_tag] = len(tag_to_idx)
          for char in word:
              if char not in char_to_idx.keys():
                  char_to_idx[char] = len(char_to_idx)

  return word_to_idx,tag_to_idx,char_to_idx

def word_to_ix(word, ix):
    return torch.tensor(ix[word], dtype = torch.long)

def char_to_ix(char, ix):
    return torch.tensor(ix[char], dtype= torch.long)

def tag_to_ix(tag, ix):
    return torch.tensor(ix[tag], dtype= torch.long)

def sequence_to_idx(sequence, ix):

    return torch.tensor([ix.get(s,ix["UNKNOWN_TOKEN"] )for s in sequence], dtype=torch.long)

class DualLSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim, char_embedding_dim, char_hidden_dim, word_vocab_size, char_vocab_size, tag_vocab_size):
        super(DualLSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        self.lstm = nn.LSTM(word_embedding_dim + char_hidden_dim, word_hidden_dim)
        self.hidden2tag = nn.Linear(word_hidden_dim, tag_vocab_size)

    def forward(self, sentence, words):
        embeds = self.word_embedding(sentence)
        char_hidden_final = []
        for word in words:
            char_embeds = self.char_embedding(word)
            _, (char_hidden, char_cell_state) = self.char_lstm(char_embeds.view(len(word), 1, -1))
            word_char_hidden_state = char_hidden.view(-1)
            char_hidden_final.append(word_char_hidden_state)
        char_hidden_final = torch.stack(tuple(char_hidden_final))

        combined = torch.cat((embeds, char_hidden_final), 1)

        lstm_out, _ = self.lstm(combined.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))

        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def accuracy(test,predicted):
  total_words = 0
  correct_predictions = 0

  for i, sentence_tag in enumerate(test):
      predicted_tags = predicted[i]
      for j, (word, correct_tag) in enumerate(sentence_tag):
          predicted_tag = predicted_tags[j][1]  # Get the predicted tag for the word
          if predicted_tag == correct_tag:
              correct_predictions += 1
          total_words += 1

  accuracy = correct_predictions / total_words
  return accuracy

def predict(test, the_model):
  predicted=[]
  for sentence_tag in test:
    seq=[]
    for word_tag in sentence_tag:
      seq.append(word_tag[0])
    with torch.no_grad():
        words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in seq]
        sentence = torch.tensor(sequence_to_idx(seq, word_to_idx), dtype=torch.long).to(device)

        tag_scores = the_model(sentence, words)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_idx.items():
                if indices[i] == value:
                    ret.append((seq[i], key))

        predicted.append(ret)
  return predicted

if __name__=='__main__':
  train_x,train_y=load_data('/content/drive/MyDrive/data_NLP_A2/en_atis-ud-train.conllu')
  test_x,test_y=load_data('/content/drive/MyDrive/data_NLP_A2/en_atis-ud-test.conllu')
  dev_x,dev_y=load_data('/content/drive/MyDrive/data_NLP_A2/en_atis-ud-dev.conllu')



  train=data_preparation(train_x,train_y)
  test=data_preparation(test_x,test_y)
  dev=data_preparation(dev_x,dev_y)
  all_data=train+test+dev



  word_to_idx,tag_to_idx,char_to_idx=calculate_index(all_data)
  word_to_idx["UNKNOWN_TOKEN"] = len(word_to_idx)
  char_to_idx["UNKNOWN_TOKEN"] = len(char_to_idx)
  word_vocab_size = len(word_to_idx)
  tag_to_idx["UNKNOWN_TOKEN"] = 'Noun'
  tag_vocab_size = len(tag_to_idx)
  char_vocab_size = len(char_to_idx)


  WORD_EMBEDDING_DIM = 1024
  CHAR_EMBEDDING_DIM = 128
  WORD_HIDDEN_DIM = 1024
  CHAR_HIDDEN_DIM = 1024
  EPOCHS = 10

  saved=False

  if saved==True:
    the_model = DualLSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, word_vocab_size, char_vocab_size, tag_vocab_size)
    the_model.load_state_dict(torch.load("/content/drive/MyDrive/NLP/A_2/LSTM/model.pth",map_location=torch.device('cpu')))
    the_model.eval()

    device = torch.device("cpu")
    # the_model.cuda()

    # predicted_tags=predict(test,the_model)
    # acc_score=accuracy(test,predicted_tags)
    # print("Accuracy:", acc_score*100)

    input_sentence=input('Enter the sentence: ')
    seq=input_sentence.split()
    with torch.no_grad():
        words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in seq]
        sentence = torch.tensor(sequence_to_idx(seq, word_to_idx), dtype=torch.long).to(device)

        tag_scores = the_model(sentence, words)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_idx.items():
                if indices[i] == value:
                    ret.append((seq[i], key))

    for word_tag in ret:
      print(word_tag[0]+'\t'+word_tag[1])
      print('\n')


  else:
    model = DualLSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM, word_vocab_size, char_vocab_size, tag_vocab_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        model.cuda()

    # Define the loss function as the Negative Log Likelihood loss (NLLLoss)
    loss_function = nn.NLLLoss()

    # We will be using a simple SGD optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # The test sentence
    seq = " what is the cost of a round".split()
    print("Running a check on the model before training.\nSentences:\n{}".format(" ".join(seq)))
    with torch.no_grad():
        words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in seq]
        sentence = torch.tensor(sequence_to_idx(seq, word_to_idx), dtype=torch.long).to(device)

        tag_scores = model(sentence, words)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_to_idx.items():
                if indices[i] == value:
                    ret.append((seq[i], key))
        print(ret)
    # Training start
    print("Training Started")
    accuracy_list = []
    loss_list = []
    interval = round(len(train) / 100.)
    epochs = EPOCHS
    e_interval = round(epochs / 1)
    for epoch in range(epochs):
        print(epoch)
        acc = 0 #to keep track of accuracy
        loss = 0 # To keep track of the loss value
        i = 0
        print(len(train))
        for sentence_tag in train:
            
            i += 1
            words = [torch.tensor(sequence_to_idx(s[0], char_to_idx), dtype=torch.long).to(device) for s in sentence_tag]
            sentence = [s[0] for s in sentence_tag]
            sentence = torch.tensor(sequence_to_idx(sentence, word_to_idx), dtype=torch.long).to(device)
            targets = [s[1] for s in sentence_tag]
            targets = torch.tensor(sequence_to_idx(targets, tag_to_idx), dtype=torch.long).to(device)

            model.zero_grad()

            tag_scores = model(sentence, words)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            _, indices = torch.max(tag_scores, 1)
    #         print(indices == targets)
            acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))
            if i % interval == 0:
                print("Epoch {} Running;\t{}% Complete".format(epoch + 1, i / interval), end = "\r", flush = True)
        loss = loss / len(train)
        acc = acc / len(train)
        loss_list.append(float(loss))
        accuracy_list.append(float(acc))
        if (epoch + 1) % e_interval == 0:
            print("Epoch {} Completed,\tLoss {}\tAccuracy: {}".format(epoch + 1, np.mean(loss_list[-e_interval:]), np.mean(accuracy_list[-e_interval:])))
    # accuracy_list,loss_list=train(EPOCHS)
    torch.save(model.state_dict(), "/content/drive/MyDrive/NLP/A_2/LSTM/model.pth")


    import matplotlib.pyplot as plt
    plt.plot(accuracy_list, c="red", label ="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    plt.savefig("/content/drive/MyDrive/NLP/A_2/LSTM/accuracy.png")

    plt.plot(loss_list, c="blue", label ="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    plt.savefig('/content/drive/MyDrive/NLP/A_2/LSTM/loss.png')
