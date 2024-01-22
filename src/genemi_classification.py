import tqdm
import keras
import numpy as np
import pandas as pd
import google.generativeai as genai
from google.api_core import retry
import matplotlib.pyplot as plt
from keras import layers
import sklearn.metrics as skmetrics
from sklearn.model_selection import train_test_split
from function import get_API_key

key_file = '/Users/namgyulee/Personal_Project/News_Article_Classification/api-key.txt'
api_key_instance = get_API_key(key_file, 8)
API_key = api_key_instance.get_api_key(8)

genai.configure(api_key=API_key)

df = pd.read_csv('/Users/namgyulee/Personal_Project/News_Article_Classification/Data/Genemi_data.csv')
df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

tqdm.pandas()

def make_embed_text_fn(model):

  @retry.Retry(timeout=300.0)
  def embed_fn(text: str) -> list[float]:
    # Set the task_type to CLASSIFICATION.
    embedding = genai.embed_content(model=model,
                                    content=text,
                                    task_type="classification")
    return embedding['embedding']

  return embed_fn

def create_embeddings(model, df):
  df['Embeddings'] = df['Text'].progress_apply(make_embed_text_fn(model))
  return df

model = 'models/embedding-001'

df_train = create_embeddings(model, df_train)

df_test = create_embeddings(model, df_test)

def build_classification_model(input_size: int, num_classes: int) -> keras.Model:
  inputs = x = keras.Input(input_size)
  x = layers.Dense(input_size, activation='relu')(x)
  x = layers.Dense(num_classes, activation='sigmoid')(x)
  return keras.Model(inputs=[inputs], outputs=x)

# Derive the embedding size from the first training element.
embedding_size = len(df_train['Embeddings'].iloc[0])

# Give your model a different name, as you have already used the variable name 'model'
classifier = build_classification_model(embedding_size, len(df_train))
classifier.summary()

classifier.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   optimizer = keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])

NUM_EPOCHS = 40
BATCH_SIZE = 32

# Split the x and y components of the train and validation subsets.
y_train = df_train['Cluster Label']
x_train = np.stack(df_train['Embeddings'])
y_val = df_test['Cluster Label']
x_val = np.stack(df_test['Embeddings'])

# Train the model for the desired number of epochs.
callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

history = classifier.fit(x=x_train,
                         y=y_train,
                         validation_data=(x_val, y_val),
                         callbacks=[callback],
                         batch_size=BATCH_SIZE,
                         epochs=NUM_EPOCHS,)
# Result: loss: 0.1149 - accuracy: 0.9850 - val_loss: 0.5434 - val_accuracy: 0.8100

classifier.evaluate(x=x_val, y=y_val, return_dict=True)
# Result: loss: 0.5434 - accuracy: 0.8100 {'loss': 0.5433962345123291, 'accuracy': 0.8100000023841858}

def plot_history(history):
  """
    Plotting training and validation learning curves.

    Args:
      history: model history with all the metric measures
  """
  fig, (ax1, ax2) = plt.subplots(1,2)
  fig.set_size_inches(20, 8)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()
  filename = "Genemi Classifier performance.png"
  plt.savefig(filename)

plot_history(history)

y_hat = classifier.predict(x=x_val)
y_hat = np.argmax(y_hat, axis=1)

labels_dict = dict(zip(df_test['ANN Label'], df_test['Cluster Label']))
labels_dict

# Visualize how different between ANN and Genemi cluster model
cm = skmetrics.confusion_matrix(y_val, y_hat)
disp = skmetrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels_dict.keys())
disp.plot(xticks_rotation='vertical')
plt.title('Confusion matrix for newsgroup test dataset');
plt.grid(False)
filename = "Genemi Confusion matrix.png"
plt.savefig(filename)

