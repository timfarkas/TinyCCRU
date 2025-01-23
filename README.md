# CCRU Procedural Generator

Welcome to the **CCRU Procedural Generator**, a project inspired by the cryptic and avant-garde style of the Cybernetic Culture Research Unit (CCRU). This generator utilizes a transformer architecture, specifically drawing from Andrej Karpathy's "GPT from scratch" tutorial. It trains a Transformer to procedurally produce text that echoes the enigmatic, fragmented narratives characteristic of CCRU's work drawing from themes of accelerationism, AI, cosmic horror, techno-capitalism, memetics, and the occult.

![tinyCCRU in action](https://video.twimg.com/ext_tw_video/1882391152986329089/pu/vid/avc1/1280x720/KcM_KVytAPRD1B5K.mp4?tag=12)
[X Video Link](https://x.com/FarkasTim/status/1882392934588489959)

## Features

- **Transformer Architecture**: Utilizes a 6-block multi-head self-attention transformer model to generate impressionist CCRU text sequences.
- **Real-time Generation**: Capable of generating text in real-time, simulating the continuous flow of CCRU's thought processes.
- **Customizable Parameters**: Adjust hyperparameters such as batch size, block size, and learning rate to fine-tune the generation process.

## Installation

To get started, clone the repository and install the required dependencies: `pip install -r requirements.txt`


#### Generate text
To generate text:
1. Run generate.py to procedurally, infinitely generate CCRU text and lean back and enjoy the vibes of cyber-occult techno-acceleration!

#### Training
To train a new model:
1. delete final_model.pth, 
2. edit train_transformer.py:
    - set generate_only to False 
    - set load_model to False
    - configure other hyperparameters
3. run train_transformer.py and watch it converge!
    - on my home cluster's RTX4090 setup, 5000 iterations took about 15 mins   

