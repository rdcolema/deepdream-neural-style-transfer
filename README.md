# Deepdream / Neural Style Transfer
Implementation of Google Deepdream w/ extra style transfer scripts

Based on https://github.com/google/deepdream and https://github.com/fzliu/style-transfer

Requires caffe and certain pre-trained models (bvlc_googlenet and vgg16)

<h2>Examples: </h2>

<h3>Deepdream</h3>
<p align="center">
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/dog_birthday_party.jpg" width="50%"/>
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/space_turkey.jpg" width="30%"/>
</p>
<h3>Style Transfer</h3>
<p>
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/l_van_gogh.jpg" width="30%"/>
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/emlets.jpg" width="30%"/>
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/da_lucy.jpg" width="30%"/>
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/mona_picasso.jpg" width="40%"/>
<img src="https://raw.githubusercontent.com/rdcolema/deepdream-neural-style-transfer/master/examples/psych_pearl.jpg" width="40%"/>
</p>

<h3>Installation</h3>

Installing caffe on Windows isn't easy, but I ended up having some success using a virtual machine and following the instructions from <a href="http://www.alanzucconi.com/2016/05/25/generating-deep-dreams/">this tutorial</a>. The hardest part here is getting the right paths set in your .bashrc file (as far as I can tell you can skip the caffe.conf step if you get the .bashrc paths right). 

For the style transfer scripts, you also might need to go out and download some models on your own, which I got from <a href="https://github.com/BVLC/caffe/wiki/Model-Zoo">here</a>. VGG16 seems to work best for me. 
