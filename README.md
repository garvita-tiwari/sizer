# SIZER: A DATASET AND MODEL FOR PARSING 3D CLOTHING AND LEARNING SIZE SENSITIVE 3D CLOTHING

Code and model for SIZER: A DATASET AND MODEL FOR PARSING 3D CLOTHING AND LEARNING SIZE SENSITIVE 3D CLOTHING, ECCV 2020(Oral)

  - [Website](https://virtualhumans.mpi-inf.mpg.de/sizer/) 
  - [Dataset](https://nextcloud.mpi-klsb.mpg.de/index.php/s/nx6wK6BJFZCTF8C) 
  - [Pre-trained model](https://nextcloud.mpi-klsb.mpg.de/index.php/s/nx6wK6BJFZCTF8C) :Coming soon

### Pre-requistes
[MPI mesh library](https://github.com/MPI-IS/mesh)
[Kaolin](https://github.com/NVIDIAGameWorks/kaolin)
Tested on Pytorch 1.4, cuda 10.1, python 3.6.10

### Data path:
    set DATA_DIR='<downloaded dataset path>/training_data' in utils/global_var.py 

### Training ParserNet:
    python trainer/parsernet.py --log_dir <log_dir_path> - 

### Evaluating ParserNet:
    python trainer/parsernet_eval.py --log_dir <log_dir_path> -

### Training SizerNet:
    python trainer/sizernet.py --log_dir <log_dir_path> - 

### Evaluating SizerNet:
    python trainer/sizernet_eval.py --log_dir <log_dir_path> -

### Evaluation script for other datasets:
    coming soon..
    
### Citation:
    @inproceedings{tiwari20sizer,
    title = {SIZER: A Dataset and Model for Parsing 3D Clothing and Learning Size Sensitive 3D Clothing},
        author = {Tiwari, Garvita and Bhatnagar, Bharat Lal and Tung, Tony and Pons-Moll, Gerard},
        booktitle = {European Conference on Computer Vision ({ECCV})},
        month = {August},
        organization = {{Springer}},
        year = {2020},
        }
        
### Acknowledgements:
    Thanks to Chaitanya Patel for pytorch implementation of SMPL4Garment (Link: https://github.com/chaitanya100100/TailorNet)
