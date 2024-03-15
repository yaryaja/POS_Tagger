import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from conllu import parse_incr
from collections import defaultdict

START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# Load data from file
def load_data(file_loc):
    data_file = open(file_loc, "r", encoding="utf-8")
    X_sentence = []
    Y_sentence = []

    for sentence in parse_incr(data_file):
        X_sentence.append([sentence.metadata['text']])
        temp = []
        for i in range(len(sentence.metadata['text'].split())):
            temp.append(sentence[i]['upos'])
        Y_sentence.append(temp)
    return X_sentence, Y_sentence



def data_preparation(x, y, left_context,right_context):
    data = []

    for sentence, tags in zip(x, y):
        size=len(y)
        sentence = sentence[0].split()
        sentence = [START_TOKEN] * left_context + sentence + [END_TOKEN] * right_context
        for i in range(left_context, len(sentence) - right_context):
            context = sentence[i - left_context:i + right_context + 1]
            target_tag = tags[i - left_context]
            word_tag_pair = (context, target_tag)  # Pair of context tokens and corresponding tag
            data.append(word_tag_pair)
    return data

# Calculate index mappings for words, tags, and characters
def calculate_index(all_data):
    word_to_idx = defaultdict(lambda: len(word_to_idx))
    tag_to_idx = defaultdict(lambda: len(tag_to_idx))
    char_to_idx = defaultdict(lambda: len(char_to_idx))
    for sentence in all_data:
        words, pos_tag = sentence
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word]=len(word_to_idx)


        tag_to_idx[pos_tag]
    return word_to_idx, tag_to_idx

# Convert sequence to indices

def sequence_to_idx(sequence, ix):
    return torch.tensor([ix.get(s,ix[UNKNOWN_TOKEN])for s in sequence], dtype=torch.long)

# Define Feed-Forward Neural Network model
class FFNNTagger(nn.Module):
    def __init__(self, word_embedding_dim, context_size, word_vocab_size, tag_vocab_size):
        super(FFNNTagger, self).__init__()
        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.hidden1 = nn.Linear((context_size) * word_embedding_dim, 512)  # Adjust input dimension
        self.hidden2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, tag_vocab_size)
        self.context_size = context_size

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)

        out = F.relu(self.hidden1(embeds))
        out = F.relu(self.hidden2(out))
        out = self.out(out)
        tag_scores = F.log_softmax(out, dim=1)
        return tag_scores

# Compute accuracy
def accuracy(test, predicted):
    total_words = 0
    correct_predictions = 0

    for sentence_tag in zip(test, predicted):
        predicted_tags = sentence_tag[1]
        for (word, correct_tag), (_, predicted_tag) in zip(sentence_tag[0], predicted_tags):
            if predicted_tag == correct_tag:
                correct_predictions += 1
            total_words += 1

    accuracy = correct_predictions / total_words
    return accuracy

# Make predictions
def predict(test, the_model):

    for sentence_tag in train:
        predicted=[]
        sentence, targets = sentence_tag
        with torch.no_grad():
            sentence = torch.tensor(sequence_to_idx(sentence, word_to_idx), dtype=torch.long).to(device)

            targets = torch.tensor(tag_to_idx[targets], dtype=torch.long).to(device)



            tag_scores = model(sentence)

    return predicted

if __name__ == '__main__':
    # Load data
    train_x,train_y=load_data('/content/drive/MyDrive/data_NLP_A2/en_atis-ud-train.conllu')
    test_x,test_y=load_data('/content/drive/MyDrive/data_NLP_A2/en_atis-ud-test.conllu')
    dev_x,dev_y=load_data('/content/drive/MyDrive/data_NLP_A2/en_atis-ud-dev.conllu')


    # Define context size (p previous tokens and s successive tokens)
    p = 2  # Number of previous tokens
    s = 2  # Number of successive tokens
    context_size = p + s + 1  # Total context size including current token

    # Prepare data
    train = data_preparation(train_x, train_y, s,p)
    test = data_preparation(test_x, test_y, s,p)
    dev = data_preparation(dev_x, dev_y, s,p)
    all_data = train + test + dev

    # # Calculate indices
    words_to_idx, tag_to_idx= calculate_index(all_data)
    words_to_idx[UNKNOWN_TOKEN] = len(words_to_idx)
    word_vocab_size = len(words_to_idx)
    tag_vocab_size = len(tag_to_idx)
    # char_vocab_size = len(char_to_idx)

    # # Model parameters
    WORD_EMBEDDING_DIM = 1024
    EPOCHS = 2
    
    ### device set up
    device=torch.device("cpu")
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda:0" if use_cuda else "cpu")
    
    
    saved=True
    if saved:
      device=torch.device("cpu")
      the_model = FFNNTagger(WORD_EMBEDDING_DIM, context_size, word_vocab_size, tag_vocab_size)
      the_model.load_state_dict(torch.load('/content/drive/MyDrive/NLP/A_2/FFNN/model.pth', map_location=torch.device('cpu')))
      the_model.eval()

      input_sentence = input('Enter the sentence: ')
      seq = input_sentence.split()
      print(seq)

      with torch.no_grad():
          sentence = seq
          sentence = [START_TOKEN] * p + sentence + [END_TOKEN] * s
          ret = []
          for i in range(p, len(sentence) - s):
              context = sentence[i - p:i + s + 1]
              input_sentence = torch.tensor(sequence_to_idx(context, words_to_idx), dtype=torch.long).to(device)

              tag_scores = the_model(input_sentence).to(device)
              value, indices = torch.max(tag_scores, 1)
              for key, value in tag_to_idx.items():
                  if indices == value:
                      ret.append((seq[i - p], key))
                      print(seq[i - p], ":", key)

                          
    else:
      model = FFNNTagger(WORD_EMBEDDING_DIM, context_size, word_vocab_size, tag_vocab_size).to(device)

      loss_function = nn.NLLLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.0001)

      seq = " what is the cost of a round".split()
      print("Running a check on the model before training.\nSentences:\n{}".format(" ".join(seq)))

      with torch.no_grad():
          sentence = seq
          sentence = [START_TOKEN] * p + sentence + [END_TOKEN] * s
          ret = []
          for i in range(p, len(sentence) - s):
              context = sentence[i - p:i + s + 1]
              input_sentence = torch.tensor(sequence_to_idx(context, words_to_idx), dtype=torch.long).to(device)

              tag_scores = model(input_sentence).to(device)
              value, indices = torch.max(tag_scores, 1)
              for key, value in tag_to_idx.items():
                  if indices == value:
                      ret.append((seq[i - p], key))
                      print(seq[i - p], ":", key)

      print("Training Started")
      accuracy_list = []
      loss_list = []
      interval = round(len(train) / 100.)
      epochs = EPOCHS
      e_interval = round(epochs / 1.)
      for epoch in range(EPOCHS):
          acc = 0
          loss = 0
          i = 0
          for sentence_tag in train:
              i += 1
              sentence, targets = sentence_tag
              sentence = torch.tensor(sequence_to_idx(sentence, words_to_idx), dtype=torch.long).to(device)
              targets = torch.tensor(tag_to_idx[targets], dtype=torch.long).to(device)
              model.zero_grad()
              tag_scores = model(sentence)
              loss = loss_function(tag_scores, torch.tensor([targets]).to(device))
              loss.backward()
              optimizer.step()
              loss += loss.item()
              _, indices = torch.max(tag_scores, 1)
              acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))
              if i % interval == 0:
                  print("Epoch {} Running;\t{}% Complete".format(epoch + 1, i / interval), end="\r", flush=True)

          loss = loss / len(train)
          acc = acc / len(train)
          loss_list.append(float(loss))
          accuracy_list.append(float(acc))
          if (epoch + 1) % e_interval == 0:
              print("Epoch {} Completed,\tLoss {}\tAccuracy: {}".format(epoch + 1, np.mean(loss_list[-e_interval:]), np.mean(accuracy_list[-e_interval:])))
      torch.save(model.state_dict(), "/content/drive/MyDrive/NLP/A_2/FFNN/model.pth")

      import matplotlib.pyplot as plt
      plt.plot(accuracy_list, c="red", label="Accuracy")
      plt.xlabel("epochs")
      plt.ylabel("value")
      plt.legend()
      plt.show()
      plt.savefig('/content/drive/MyDrive/NLP/A_2/FFNN/accuracy.png')

      plt.plot(loss_list, c="blue", label="Loss")
      plt.xlabel("Epochs")
      plt.ylabel("Value")
      plt.legend()
      plt.show()
      plt.savefig('/content/drive/MyDrive/NLP/A_2/FFNN/loss.png')
