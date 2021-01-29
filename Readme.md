# Deepr
A Deep Reinforcement Learning(DRL) pipeline for Rust.

The goal is to simplify the implementation process of developing a DRL models, abstracting away the model implementation. Currently thist crate supports [Softmax Deep Double Double Deterministic Policy Gradients (SD3](https://arxiv.org/pdf/2010.09177v1.pdf) with pytorch.

## TODO
- Docs & Examples
- Tensorflow Support   
Right we use pytorch for our models, it would be nice to have an feature flag to optionally use tensorflow instead.
- Additional Models   
Right now we only support [SD3](https://arxiv.org/pdf/2010.09177v1.pdf). It would be nice to include different optional models like TRPO, SAC, HER. Maybe also include some model based approaches like MBVE but that might be suited for a different crate. 
