import os 

datasets = ['cifar10','svhn']
optims = ['no_optimisation','hpsklearn']

dc_path = "C:\\Users\\alan\\Desktop\\experiments\\dcgan_imagenet_64\\netD_epoch_24.pth"
wgan_path = "C:\\Users\\alan\\Desktop\\experiments\\wdcgan_imagenet_64\\netD_epoch_24.pth"
wgangp_path = "C:\\Users\\alan\\Desktop\\experiments\\wgangp_imagenet_32\\netD_epoch_24.pth"
base = 'python model_evaluator.py --disc_path '
#dcgans
# for i in datasets:
#     for j in optims:
#         os.system(base+dc_path+" --gan_type dcgan"+" --dataset "+i+" --optimise "+j) hny   
    
# #wgans
# for i in datasets:
#     for j in optims:
#         os.system(base+wgan_path+" --gan_type dcgan --wgan True "+" --dataset "+i+" --optimise "+j)

#wgangp
for i in datasets:
    for j in optims:
        os.system(base+wgangp_path+" --gan_type dcgan --wgan True --gradient_penalty True --image_size 32"+" --dataset "+i+" --optimise "+j)
    
  