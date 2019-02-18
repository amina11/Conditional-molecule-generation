## How to run the code ##

* Download the dataset from the following link

https://www.dropbox.com/sh/621ufmvqgg5h2d8/AAARWPpuADNfPx8eu9E8y-rha?dl=0

Then run prepare_data.sh, modify the directory where to read and write, by default it will seperate train and testset (5000), save both train and testset under the folder './data/data_278'

* Train supervised vae for 500 epochs     ./supervised_vae/run_main.sh

* Train the regressor on the pretrained supervised vae decoder output   ./regressor/run_regressor.sh

* Train the final model, initialize the vae part with the pretrained supervised vae model at epoch-40.model, regressor with the previously trained best regressor ./run_main.sh


