# Attention Is All You Need — Intro & Section 2 Summary

## Problem Definition
The article explores the problems of RNNs and Convolutional neural networks for longer sequences, where, as author explained,
the sequential computation leads to the problem of increasing difficulty for RNNs and CNNs to model long-range dependencies. And even regarding
the optimization of the model combined with improvements in computational efficiency for these networks, the problem of memory dedication
remains unsolved.

## Key Idea
In order to solve all the problems with memory at once, author has introduced a new structure of neural network based entirely on the 
attention mechanisms - Transformers. His idea was to reduce the number of steps required to reach a certain point on neural network by showcasing dependencies
between the nodes. His idea was to connect the dependent nodes directly, allowing to significantly reduce the computational resources and training time needed to reach a similar output within a large network

## Architecture Highlights

### Parallelizable 
Transformers allow completing multiple operations like matrix multiplications at once by splitting load across different threads, that reduces the time required to complete the task.

### Uses attention instead of recurrence
As discussed earlier, use of dependencies makes it unnecessary for the model to be sequential. In sequential model, by aligning the positions to steps in computation time, the models like RNN create
a sequence of states h(t), as a function of the previous hidden state h(t−1) and the input for position t. This process can significantly increase the amount of memory used for the task as instead of one jump, model
would have a vanishing and exploding gradient problem.

### Captures long-range dependencies
Idea of attention allows to skip the unnecessary nodes and "jump" directly, even if neurons are located far apart. 

## Simple Diagram in ASCII

RNN (sequential):   Word1 -> Word2 -> Word3 -> Output


Transformer (parallel):

(Word1  Word2  Word3)
   |      |      |
 ------Attention------
          |
        Output

## What I Don’t Fully Understand Yet
1. In practice, does attention replace or modify existing connections in the network?
2. How would you implement it in the code?
