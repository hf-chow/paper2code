This implementation of AlexNet follows the "One Werid Trick" version.
Arcitecture as follows:

[data]
type=data
dataIdx=0

[labvec]
type=data
dataIdx=1

[conv1]
type=conv
inputs=data
channels=3
filters=64
padding=0
stride=4
filterSize=11
initW=0.01
sumWidth=4
sharedBiases=1
gpu=0

[rnorm1]
type=cmrnorm
inputs=conv1
channels=64
size=5

[pool1]
type=pool
pool=max
inputs=rnorm1
sizeX=3
stride=2
channels=64
neuron=relu

[conv2]
type=conv
inputs=pool1
filters=192
padding=2
stride=1
filterSize=5
channels=64
initW=0.01
initB=1
sumWidth=3
sharedBiases=1
neuron=relu

[rnorm2]
type=cmrnorm
inputs=conv2
channels=192
size=5

[pool2]
type=pool
pool=max
inputs=rnorm2
sizeX=3
stride=2
channels=192

[conv3]
type=conv
inputs=pool2
filters=384
padding=1
stride=1
filterSize=3
channels=192
initW=0.03
sumWidth=3
sharedBiases=1
neuron=relu

[conv4]
type=conv
inputs=conv3
filters=256
padding=1
stride=1
filterSize=3
channels=384
neuron=relu
initW=0.03
initB=1
sumWidth=3
sharedBiases=1

[conv5]
type=conv
inputs=conv4
filters=256
padding=1
stride=1
filterSize=3
channels=256
initW=0.03
initB=1
sumWidth=3

[pool3]
type=pool
pool=max
inputs=conv5
sizeX=3
stride=2
channels=256
neuron=relu

[fc4096a]
type=fc
inputs=pool3
outputs=4096
initW=0.01
initB=1
neuron=relu
gpu=0

[dropout1]
type=dropout2
inputs=fc4096a

[fc4096b]
type=fc
inputs=dropout1
outputs=4096
initW=0.01
initB=1
neuron=relu
gpu=0

[dropout2]
type=dropout2
inputs=fc4096b

[fc1000]
type=fc
outputs=1000
inputs=dropout2
initW=0.01
initB=-7
gpu=0

[probs]
type=softmax
inputs=fc1000

[logprob]
type=cost.logr
