# SIZER: A DATASET AND MODEL FOR PARSING 3D CLOTHING AND LEARNING SIZE SENSITIVE 3D CLOTHING


#### For SIZER Dataset: 
    https://github.com/garvita-tiwari/sizer_dataset
    
    
Code and model for SIZER: A DATASET AND MODEL FOR PARSING 3D CLOTHING AND LEARNING SIZE SENSITIVE 3D CLOTHING, ECCV 2020(Oral)

  - [Website](https://virtualhumans.mpi-inf.mpg.de/sizer/) 
  - [Dataset and Model](https://nextcloud.mpi-klsb.mpg.de/index.php/s/nx6wK6BJFZCTF8C)
    (Drop a mail to : gtiwari@mpi-inf.mpg.de(or garvita.tiwari@uni-tuebingen.de) for getting access to dataset)
  - [Fill this form for dataset](https://docs.google.com/forms/d/e/1FAIpQLSddBep3Eif1gI-6IhaZybBDoR-_H_QW1NST0JV5vviauvPNTA/viewform?usp=sf_link) 
  

###SIZER Dataset:
    https://github.com/garvita-tiwari/sizer_dataset

### Data path:
    set DATA_DIR='<downloaded dataset path>/training_data' in utils/global_var.py 

### Conda Environment:
    conda create --name sizer --file requirements.txt
    conda activate sizer


### Training ParserNet:
    python trainer.py --config=<configs/parser_default.yaml>

### Evaluating ParserNet:
    python generate.py --config=<configs/parser_default.yaml>

### Training SizerNet:
    python  trainer.py --config=<configs/sizer_default.yaml>

### Evaluating SizerNet:
    python generate.py --config=<configs/sizer_default.yaml>


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
