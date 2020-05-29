
# Five PyTorch functions you should know

Here's the table of contents:

1. TOC
{:toc}

### Introducing PyTorch tensor operations

PyTorch is a widely used, open-source deep learning platform which has been developed by the Facebook AI Research (FAIR) team, back in early 2017.

Tensor operations are at the core of everything we do in Deep Learning and PyTorch is one of the main Python libraries to facilitate tensor operations. 

This library allows us to use the usual arithmetic operations we use for numbers but applied to tensors. Also Pytorch also let us automatically compute the derivative of tensor operations which is very useful for Machine Learning and Deep Learning.

In this notebook I will introduce you to 5 useful functions to deal with tensors:

- function 1: torch.min() and torch.max()
- function 2: torch.tanh(input, out=None) → Tensor
- function 3: torch.item() → number and torch.tolist()
- function 4: torch.get_device() -> Device ordinal (Integer)
- function 5: torch.view(*shape) → Tensor

### Importing PyTorch and the necessary modules

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=2" title="Jovian Viewer" height="177" width="800" frameborder="0" scrolling="auto" border-radius= "10px"></iframe>
</div>

Here below I will describe in details how to use the functions.

## Function 1 - torch.min() and torch.max()

The two min and max functions are similar:  

#### torch.min(input) → Tensor  

Returns the minimum value of all elements in the input tensor.
`torch.min(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)`  
Returns a namedtuple (values, indices) where values is the minimum value of each row of the input tensor in the given dimension dim. And indices is the index location of each minimum value found (argmin).
If keepdim is True, the output tensors are of the same size as input except in the dimension dim where they are of size 1. 
<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=5" title="Jovian Viewer" height="235" width="800" frameborder="0" scrolling="auto"></iframe>

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=6" title="Jovian Viewer" height="143" width="800" frameborder="0" scrolling="auto"></iframe>
</div>
In the example above a tensor of shape 2, 4 (2 rows x 4 columns) has been reduced to a single row with the command `torch.min(a, 0)`, with `0` meaning the rows will be my axe, so it will return one row and the indices give me which row has the minimum, the first or the second in this case.

#### torch.max(input) → Tensor  
the max function below is similar to min therefore:  

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=8" title="Jovian Viewer" height="185" width="800" frameborder="0" scrolling="auto"></iframe>

<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=9" title="Jovian Viewer" height="143" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

#### The following example will give an error
<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=11" title="Jovian Viewer" height="null" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

This is because both min() and max() take a certain amount of parameters and at least one, which is the input tensor. The format is `max(Tensor input, int dim, bool keepdim)`, so the third parameter if present needs to be a `Boolean`. 

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=13" title="Jovian Viewer" height="159" width="800" frameborder="0" scrolling="auto"></iframe>
</div>



<!--

Here's the table of contents:

1. TOC
{:toc}

## Basic setup

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-filename.md`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `filename` is whatever file name you choose, to remind yourself what this post is about. `.md` is the file extension for markdown files.

The first line of the file should start with a single hash character, then a space, then your title. This is how you create a "*level 1 heading*" in markdown. Then you can create level 2, 3, etc headings as you wish but repeating the hash character, such as you see in the line `## File names` above.

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
