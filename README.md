Quasi-RNN for Lasagne
=====================
Quasi-Recurrent Neural Networks were introduced at ICLR 2017 by Bradbury et al., see <https://arxiv.org/abs/1611.01576>. We provide basic Quasi-RNN layers that can be used together with the Lasagne framework. The implementation is **not** necessarily optimized for fast parallel execution, as explained in the paper, and is kept as basic as possible. A language modeling example is provided in the accompanying notebook.

Pooling
-------
Three pooling modes can be used, as explained in the original paper. They can be activated using the `pooling` argument. The modes are `'f'`, `'fo'` and `'ifo'`.

