
# Five PyTorch functions you should know

Here's the table of contents:

1. TOC
{:toc}

### Introducing PyTorch tensor operations

PyTorch is a widely used, open-source deep learning platform which has been developed by the Facebook AI Research (FAIR) team, back in early 2017. PyTorch is a library for Python not a framework :)

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

## Function 2 - torch.tanh(input, out=None) → Tensor

The `tanh` function is often used in Deep Learning. It stays for `hyperbolic tangent` and returns a non linear output between -1 and 1. It is also used as an activation function.  

I will first show the shape of the `tanh` using matplotlib, a library for plotting graphs

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=16" title="Jovian Viewer" height="492" width="800" frameborder="0" scrolling="auto"></iframe>
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=17" title="Jovian Viewer" height="171" width="800" frameborder="0" scrolling="auto"></iframe>
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=18" title="Jovian Viewer" height="122" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

We can see that for inputs very close to zero the output is almost the same but grows rapidly but never get bigger than 1 or minus 1. 

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=20" title="Jovian Viewer" height="151" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

For very big positive or negative number the output grows asymptotically to 1 or minus 1. Sometimes we need to map the output values in between -1 to 1 like yes or no, and this is why we use activation functions like `tanh()` with Neural Networks.  


<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=22" title="Jovian Viewer" height="151" width="800" frameborder="0" scrolling="auto"></iframe>
</div>
The `tahn()` function is often used as an activator function in Deep Learning together with the `sigmoid` function and the `ReLu`. It is considered to be better than the `sigmoid` function because of a steeper curve for small values close to zero and it is also sigmoidal (s-shaped). It is a very robust function and cannot be broken easily unless we give a non-numeric input, this would be the only way to get an error message.  
The output type is `torch.float32`

## Function 3 - torch.item() → number and torch.tolist() 

This function returns the tensor as a (nested) list as a standard Python number. For scalars, a standard Python number is returned, just like with `item()`. 


<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=25" title="Jovian Viewer" height="195" width="800" frameborder="0" scrolling="auto"></iframe>
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=26" title="Jovian Viewer" height="162" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

As in the example above we see that both methods can be applied to a tensor containing a single element.

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=28" title="Jovian Viewer" height="202" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

In this case our input tensor has more than one item so we get a python (nested) list as output.

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=30" title="Jovian Viewer" height="242" width="800" frameborder="0" scrolling="auto"></iframe>
</div>
I could not find a way to break this function as long as the input tensors have a valid value. 

#### When to use

It is useful for the case when I have the output values in a tensor type and need to translate those to a pure python environment. 

## Function 4 - torch.isnan()

Returns a new tensor with boolean elements representing if each element is NaN or not.

<div style="border-radius: 10px; align: center; overflow: hidden;">  
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=34" title="Jovian Viewer" height="134" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

My output is a new tensor with the same dimension of the input containing only Boolean values. Using `isnan()` I can verify only the `nan` case. If I have an `inf`input it will not be returned true. 

<div style="border-radius: 10px; align: center; overflow: hidden;"> 
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=37" title="Jovian Viewer" height="134" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

For infinity values I need another function `isinf()`
<div style="border-radius: 10px; align: center; overflow: hidden;"> 
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=40" title="Jovian Viewer" height="null" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

# Example 3 - breaking 

I could not find an example to break this method

#### When to use
It is useful when I want to make sure that my input data doesn't contain values that could bring errors and detect `nan` or `inf` types in my dataset.

## Function 5 - torch.view(*shape) → Tensor

This function returns a new tensor with the same data as the input tensor with a different shape.

The returned tensor shares the same data and must have the same number of elements, but may have a different size.
<div style="border-radius: 10px; align: center; overflow: hidden;"> 
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=45" title="Jovian Viewer" height="230" width="800" frameborder="0" scrolling="auto"></iframe>
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=20" title="Jovian Viewer" height="151" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

This example will not work!

<div style="border-radius: 10px; align: center; overflow: hidden;"> 
<iframe src="https://jovian.ml/embed?url=https://jovian.ml/pymultitudes/01-tensor-operations/v/6&cellId=48" title="Jovian Viewer" height="280" width="800" frameborder="0" scrolling="auto"></iframe>
</div>

If my new size doesn't match the number of elements I will have an error. Because a shape of 3 by 8 has 24 elements,  I cannot match the 4x4 (16 elements) shape of my original tensor

#### When to use

A common issue with designing Neural Networks is when the output tensor of one layer is having the wrong shape to act as the input tensor to the next layer.

Sometimes we need to explicitly reshape tensors and we can use the view function to achieve this.

## Conclusion

These are just 5 functions selected from the many available in the PyTorch documentation showcasing the versatility of this library. There is still much more to discover.

## Reference Links

* Official documentation for `torch.Tensor`: https://pytorch.org/docs/stable/tensors.html
* Plotting the tanh function: https://www.geeksforgeeks.org/numpy-tanh-python/





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
