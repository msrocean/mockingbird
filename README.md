:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Reasearch purposes ONLY. :warning: :warning: :warning:

This repository contains code and data of the paper ***Mockingbird*: Defending Against Deep-Learning-Based Website Fingerprinting Attacks with Adversarial Traces**, published in **IEEE Transactions on Information Forensics and Security (TIFS)**. The preprint version of the paper is available at: [arXiv version](https://arxiv.org/abs/1902.06626). *Mockingbird* is designed to work against deep-learning-based website fingerprinting attacks. Extensive evaluation shows that *Mockingbird* is effective against both white-box and black-box attacks including a more advanced intersection attacks.

![Mockingbird Defense](./repo_images/mockingbird_arc.jpeg)


#### Reference Format
```
@article{rahman2020mockingbird,
      title={{Mockingbird:} Defending Against Deep-Learning-Based Website Fingerprinting Attacks with Adversarial Traces}, 
      author={Mohammad Saidur Rahman and Mohsen Imani and Nate Mathews and Matthew Wright},
      year={2020},
      eprint={1902.06626},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

### Dependencies & Required Packages
Ensuring all the depencies is critical. It is hard to keep all the packages updated at once. Some versions might be relaitvely old.
So we suggest the users to create a `python virtual environment` or `conda environment` and install the required packages.

Please make sure you have all the dependencies available and installed.

- NVIDIA GPU should be installed in the machine, running on CPU will significantly increase time complexity.
- Ubuntu 16.04.5/ CentOS Linux 7 
- Python3-venv/ conda
- Keras version: 2.2.4
- TensorFlow version: 1.15.0
- Numpy: 1.16.6
- Matplotlib: 2.2.5
- CUDA Version: 10.2 
- CuDNN Version: 7 
- Python Version: 2.7.5

-- Please install the required packages using the following command:
`pip install -r requirements.txt`

### Dataset
We have shared the processed data using a Google Drive. Please download the processed data from this Google Drive [URL](https://drive.google.com/drive/folders/10rdGknCtp6KF75DXRTvS-mle4wj9Q_vD?usp=sharing).
After downloading, please put the data into the `dataset` directory.



### Questions, Comments, & Feedback
Please, address any questions, comments, or feedback to the authors of the paper.
The main developers of this code are:
 
* Mohammad Saidur Rahman ([saidur.rahman@mail.rit.edu](mailto:saidur.rahman@mail.rit.edu)) 
* Mohsen Imani ([imani.moh@gmail.com](mailto:imani.moh@gmail.com))
* Nate Mathews ([nate.mathews@mail.rit.edu](mailto:nate.mathews@mail.rit.edu))
* Matthew Wright ([matthew.wright@rit.edu](mailto:matthew.wright@rit.edu))


### Acknowledgements
This material is based upon work supported in part by the **National Science Foundation (NSF)** under Grants No. **1423163**, **1722743**, **1816851**, and **1433736**.
