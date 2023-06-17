### **接口调用实例说明**

**1、搭建环境**

`conda create -n GAN_MNIST_generate python=3.8`

`conda activate GAN_MNIST_generate`

`conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch`

`pip install timm==0.3.2`

·解决使用终端打开jupyter notebook的问题（非必要）

`pip install jupyter_contrib_nbextensions`

`jupyter contrib nbextension install --user`

`pip install jupyter_nbextensions_configurator`

`jupyter nbextensions_configurator enable --user`

**2、添加库**

`pip install matplotlib`

`pip install requests`

`pip install scikit-image`

·解决运行中出现的问题（非必要）

`pip install Pillow==9.5.0`

**3、编写程序调用接口**

（1）定义所需要生成的数字，这里即为需要生成数字0~9

注：由于网络结构，不能单独生成1个数字

`input_label = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)` 

（2）获取所需生成数字的数量

`n = input_label.size()[0]`

（3）定义迭代次数，建议为25

`epoch_times = 25`

（4）实例化类对象，输入所需生成数字的数量和内容

`aigcmn = AiGcMn(n, input_label)`

（5）调用generate函数，获取输出的所需tensor对象

`res = aigcmn.generate(n, epoch_times)`