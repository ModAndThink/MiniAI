# MiniAI

# How it's work

  Actually, i haven't any python packages, so to use it, you would need to copy the script directly into your project. There is few functions.

### `sigmoid(x)`

  Useful function in AI to do simple binary choice

### `identity(x)`
  
  Return `x`

### `generateNeuralNetwork(nInput,wHidden,hHidden,nOutput,activationFunc)`

  Return a model with `nInput` of node input, `nOutput` of node output, `wHidden` of hidden layer and `hHidden` of node per hidden layer. The `activationFunc` is where you but your activation function (logic) like the sigmoid to do binary choice.

### `mutate(neural_network,mutationRange)`

  Return a mutated model of `neural_network` in range define by `mutatonRange`. You normally wouldn't have to use it.

### `generate_child(neural_network,parents,mutationRange)`

  Return a mutated model of `neural_network` mix with `parent` model attribute (note : `parent` is an array composed of the best individu of your current generation) in range define by `mutationRange`.

### `returnValue(neural_network,inputVal)`

  Using input values give in `inputVal`, return the outputs generated using the model `neural_network`.
