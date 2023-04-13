# Evolutionary optimization of convolutional ELM for remaining useful life prediction 
This is the repo for the paper "Evolutionary optimization of convolutional ELM for remaining useful life prediction " which is an extension of its previous work specified in [https://github.com/mohyunho/MOO_ELM](https://github.com/mohyunho/MOO_ELM)


## MOO CELM
<p align="center">
  <img height="250" src="/conv_elm_overview.png">
</p>

The objective of this study is to search for the best convolutional ELM, so-called conv ELM or CELM, architectures in terms of a trade-off between RUL prediction error and training time, the latter being determined by the number of trainable parameters, on the CMAPSS dataset.
<br/>
you can find the trade-off solution by running the python codes below:
```bash
python3 enas_convELM_CMAPSS.py
```
<p align="center">
  <img height="250" src="/conv_elm_params.png">
</p>



Our experimental results on the CMAPSS dataset are shown as below: (a) FD001, (b) FD002, (c) FD003, and (d) FD004.<br/>
<p align="center">
  <img height="500" src="/conv_elm_results.png">
</p>


To cite this code use
```
@article{mo2023celm,
  title={Evolutionary Optimization of Convolutional Extreme Learning Machine for Remaining Useful Life Prediction},
  author={Mo, Hyunho and Iacca, Giovanni},
  journal={SN Computer Science},
  year={2023},
  publisher={Springer},
  note={to appear},
}
```

## References
<a id="1">[1]</a> 
Hyunho Mo and Giovanni Iacca. Evolutionary optimization of convolutional extreme learning machine for remaining useful life prediction. SN Computer Science, 2023. to appear.
