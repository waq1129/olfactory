# olfactory
olfactory (multi-trial GPLVM)

The code is for paper [Learning a latent manifold of odor representations from neural responses in piriform cortex](https://proceedings.neurips.cc/paper/2018/file/17b3c7061788dbe82de5abe9f6fe22b3-Paper.pdf).

Run ``gen_syn_2d.ipynb`` to generate 2d simulated data.

Run ``demo1.ipynb`` and ``demo2.ipynb`` for multi-trial GPLVM fit to the simulated data. You will recover the latent and reconstruct the firing rates. 

``demo2.ipynb`` consists of ``demo1.ipynb`` and a second stage. The first stage (same as ``demo1.ipynb``) estimates latent and model parameters with the naive model assumption. The second stage is a fine-tune of latent and model parameters with the user-specified model assumption. 

More details can be found in the notebooks.

All required packages are included in olfactory.yml. You can install a conda env via
``conda env create -f olfactory.yml``
