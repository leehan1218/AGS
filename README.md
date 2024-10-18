# AGS network for skin lesion prediction
![Pipeline_new](https://github.com/user-attachments/assets/3f3188fd-fe19-4fef-b584-10ba3d14794d)

 we propose a new classifier incorporating the AGS network, which integrates multiple modules to enhance data and improve prediction accuracy in the classification of skin lesions. Additionally, we analyze the individual and collective contributions of these modules to evaluate their impact on the overall model performance. The AGS network, the core of this classifier, consists of three modules: Module A (Augmentation), Module G (GANs), and Module S (Segmentation).
 
 Module A applies traditional image augmentation techniques such as flipping, rotation, and color jittering. Module G uses Progressive Growing of GANs (PGGAN) to generate high-quality synthetic images, mitigating the issue of limited dataset size. Module S employs the U-Net algorithm to segment the lesion area, removing background noise such as age spots and microscope artifacts, enabling more efficient feature learning. To evaluate the AGS network, we combined it with six state-of-the-art deep learning models including GoogLeNet, DenseNet201, ResNet50, MobileNet V3, EfficientNet B0, and ViT, and tested five different configurations: (1) the base model with raw data, (2) with Module A, (3) with Modules A and G, (4) with Modules A and S, and (5) the full AGS network with all three modules. The results showed that the AGS network performed best when all modules were used together.


 




The links to the original code for each module of the proposed methodology are as follows:

Module A : https://github.com/pytorch/pytorch
Module G : https://github.com/odegeasslbc/Progressive-GAN-pytorch?tab=readme-ov-file#reference
Module S : https://github.com/usuyama/pytorch-unet?tab=readme-ov-file
