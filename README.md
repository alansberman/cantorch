# cantorch
MSC

*** Need to flesh this out ***

Pytorch implementation of the Creative Adversarial Network (++, as will extend).

```main.py``` = driver

```model.py``` = sets up and trains the GAN 

```generators.py``` = GAN generators (DCGAN/CAN/WGAN etc.)

```discriminators.py``` = GAN discriminators (DCGAN/CAN/WGAN etc.)

```ops.py``` = (Pytorch) Dataset set up and some other useful stuff

```utils.py``` = was going to store loss functions but Pytorch seems to have that covered! (Should remove)

```image_check.py``` = Little script that checks dimensions of Wikiart dataset post image resizing

To run on Floydhub and use their Tesla K80:

```floyd run --gpu --env pytorch-0.3 --data asberman/datasets/wiki256/1:/mydata "python main.py --cuda --out_folder /output"```

else:

```python main.py``` (but need to edit  ```--dataroot``` and ```--out_folder``` in ```main.py``` to reflect local dir)

