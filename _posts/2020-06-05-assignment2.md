# Predict the water temperature based on salinity


> The CalCOFI data set represents the longest (1949-present) and most complete (more than 50,000 sampling stations) time series of oceanographic and larval fish data in the world. It includes abundance data on the larvae of over 250 species of fish; larval length frequency data and egg abundance data on key commercial species; and oceanographic and plankton data. The physical, chemical, and biological data collected at regular time and space intervals quickly became valuable for documenting climatic cycles in the California Current and a range of biological responses to them. 

### Is there a relationship between water salinity & water temperature?

CalCOFI: Over 60 years of oceanographic data: Is there a relationship between water salinity & water temperature? Can you predict the water temperature based on salinity? I took the data from the Kaggle website: and ran the notebook on Kaggle with the CalCOFI data files provided. 

### Download and explore the dataset

The data will first be loaded as a Pandas dataframe. I will take the first 700 rows of data from the dataset.

```python
dataframe = pd.read_csv(DATASET_URL)
# I take the first 700 data points to examine more in detail
df = dataframe[:][:700]
```

I can now see that I have some columns which contain not numeric values, which are ID strings, and some columns and rows, which have many missing values in it (shown as 'NaN' in the dataset).

<p align="center">
 <img src="/images/CalCOFI/image1.png" width="750" title="vs code">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p> 

And I noticed browsing the data that the columns that are the most important for my model, contain some few NaN values too! 
I will remove those rows first. So I make a copy of the dataset taking away any row, which shows as NaN in the two columns: "Saltiness" and "Temperature" in the dataset ( `Salnty` and `T_degC`). 
I do a sanity check and print the number of rows left after every operation. 

```python
df = df[df['Salnty'].notna()]
print("rows are now ", len(df))
df = df[df['T_degC'].notna()]
print("rows are now ", len(df))
df.head()
```

I had a few rows with NaNs in the temperature and salinity columns which I now have taken out; I get now 675 rows.

Also, I want to have only the columns with numerical data. I am not interested in Strings and ids for this dataset.
There is a handy function for this: 

```python
df = df._get_numeric_data()
```

I will define my output column. The inputs will be all my columns except the temperature column which is my output or target:

```python
output_cols = ['T_degC']
input_cols = df.columns[df.columns!='T_degC']
```

So finally I have my pandas inputs and outputs and my dataset is looking better. You can see the first 5 rows below:

<p align="center">
 <img src="/images/CalCOFI/image2.png" width="750" title="vs code">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p> 

### Inspect relationships between the data

I am interested to observe the relationship between the data I extracted from the dataset. The inputs are the saltiness of the water and the depth and a few other variables. I want to create a model able to predict the temperature.

I make a scatter plot to see any visual relationship between the data with the `seaborn` library for python.


```python
sns.lmplot(x="Salnty", y="T_degC", data=df,
 order=2, ci=None);

sns.lmplot(x="Depthm", y="T_degC", data=df,
 order=2, ci=None);
``` 

<p align="center">
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/jovian-assignment-2/v/15&cellId=21" title="Jovian Viewer" height="465" width="800" frameborder="0" scrolling="auto"></iframe>
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/jovian-assignment-2/v/15&cellId=22" title="Jovian Viewer" height="465" width="800" frameborder="0" scrolling="auto"></iframe>
</p>

Now I can convert my pandas dataframes to numpy arrays ( df is my dataset ):

```python
inputs_array = df[input_cols].to_numpy()
targets_array = df[output_cols].to_numpy()
```

And then to torch tensors. I also need to make sure that the data type is torch.float32.

```python
dtype = torch.float32
inputs = torch.from_numpy(inputs_array).type(dtype)
targets = torch.from_numpy(targets_array).type(dtype)
```

Next, we need to create PyTorch datasets & data loaders for training & validation. We'll start by creating a TensorDataset. I will choose a number between 0.1 and 0.2 to determine the fraction of data that will be used for creating the validation set. Then use random_split to create training & validation datasets.

```python
dataset = TensorDataset(inputs, targets)
val_percent = 0.1 # between 0.1 and 0.2
num_rows = len(df)
num_cols = len(df.columns)
val_size = int(num_rows * val_percent)

train_size = num_rows - val_size
train_ds, val_ds = random_split(df,(train_size, val_size)) 
```

Pick a batch size for the data loader.

```python
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
```


### Create a Linear Regression Model

Finally, after all the preparation we create a model. It is a linear regression because we want to predict a continuous value, which is the temperature. We have quite a few inputs but clearly, the salinity and the depth are the most important for us. 

This class will initialize our model like we did in the previous notebook: 

```python
class TempModel(nn.Module):
 def __init__(self):
 super().__init__()
 self.linear = nn.Linear(input_size, output_size)
 def forward(self, xb):
 out = self.linear(xb)
 
 def training_step(self, batch):
 inputs, targets = batch 
 # Generate predictions
 out = self(inputs)
 # Calculate loss
 loss = F.l1_loss(out,targets)
 return loss
 
 def validation_step(self, batch):
 inputs, targets = batch
 # Generate predictions
 out = self(inputs)
 # Calculate loss
 loss = F.l1_loss(out,targets)
 return {'val_loss': loss.detach()}
 
 def validation_epoch_end(self, outputs):
 batch_losses = [x['val_loss'] for x in outputs]
 epoch_loss = torch.stack(batch_losses).mean() # Combine losses
 return {'val_loss': epoch_loss.item()}
 
 def epoch_end(self, epoch, result, num_epochs):
 # Print result every 20th epoch
 if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
 print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
```

We instantiate our model:

```python
model = TempModel()
```

### Train the model to fit the data

We define two more functions

```python
def evaluate(model, val_loader):
 outputs = [model.validation_step(batch) for batch in val_loader]
 return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
 history = []
 optimizer = opt_func(model.parameters(), lr)
 for epoch in range(epochs):
 # Training Phase 
 for batch in train_loader:
 loss = model.training_step(batch)
 loss.backward()
 optimizer.step()
 optimizer.zero_grad()
 # Validation phase
 result = evaluate(model, val_loader)
 model.epoch_end(epoch, result, epochs)
 history.append(result)
 return history

result = evaluate (model,val_loader)
print(result) 
```

This prints `{'val_loss': 49.7645149230957}`. No wonder it is so high since the model has been initialized at random!

We are now ready to train the model. We decide a number of epochs and a learning rate and we pass it to the functions. The result will be saved in the history so we can plot a graph at the end of the training. 

```python
epochs = 100
lr = 1e-6
history = fit(epochs, lr, model, train_loader, val_loader)
```

So now at the end, I get a validation loss of `val_loss = 1.0058`.

### Make predictions using the trained model

We define our predict function:

```python
def predict_single(input, target, model):
 inputs = input.unsqueeze(0)
 predictions = model(inputs) # fill this
 prediction = predictions[0].detach()
 print("Input:", input)
 print("Target:", target)
 print("Prediction:", prediction)
```

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/jovian-assignment-2/v/15&cellId=62" title="Jovian Viewer" height="265" width="800" frameborder="0" scrolling="auto"></iframe>

These are the results for some individual inputs. It is not bad for a first run but the accuracy is still quite low. 

### Conclusion

I never could have imagined that there is a correlation between the salinity and the temperature of the sea. I believe that the results are partially combined with the depth at which the samples were taken. The difference due to the salinity of the water is probably so small that my model would not be able to capture accurately. Also, I noticed that much of the data has quite a lot of noise, so I am not at all sure that am accurate model can even be created. 



### Resources


This is the link to the notebook on [Jovian](https://jovian.ml/pymultitudes/jovian-assignment-2/v/15)
And this is the Kaggle site where you can find the input dataset: [https://www.kaggle.com/sohier/calcofi](https://www.kaggle.com/sohier/calcofi) 

I enjoyed this exercise as part of the assignment of week two of the [freecodecamp](https://www.freecodecamp.org/) course in collaboration with [Jovian](https://jovian.ml/): [Deep Learning with PyTorch: Zero to GANs](https://jovian.ml/forum/c/pytorch-zero-to-gans/18)


<!--

Here's the table of contents:

1. TOC
{:toc}

## Basic setup

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

## Basic formatting

You can use *italics*, **bold**, `code font text`, and create [links](https://www.markdownguide.org/cheat-sheet/). Here's a footnote [^1]. Here's a horizontal rule:

---

## Lists

Here's a list:

- item 1
- item 2

And a numbered list:

1. item 1
1. item 2

## Boxes and stuff

> This is a quotation

{% include alert.html text="You can include alert boxes" %}

...and...

{% include info.html text="You can include info boxes" %}

## Images

![](/images/logo.png "fast.ai's logo")

## Code

General preformatted text:

 # Do a thing
 do_thing()

Python code and output:

```python
# Prints '2'
print(1+1)
```

 2

## Tables

| Column 1 | Column 2 |
|-|-|
| A thing | Another thing |

## Footnotes

[^1]: This is the footnote.

-->

