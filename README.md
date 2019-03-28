@[TOC](使用imgaug图像数据增强库对影像上多个Bounding Boxes进行增强)

# 简介

相较于Augmentor，imgaug具有更多的功能，比如对影像增强的同时，对keypoint, bounding box进行相应的变换。例如在目标检测的过程中，训练集包括影像及其对应的bounding box文件，在对影像增强的时候，同时解算出bounding box 相应变换的坐标生成对应的bounding box文件。

[代码](https://github.com/mickkky/XML-Augment.git)
# imgaug安装

[imgaug使用文档](https://imgaug.readthedocs.io/en/latest/index.html)

安装依赖库

```Python
pip install six numpy scipy matplotlib scikit-image opencv-python imageio
```

安装imgaug

方式一（安装github最新版本）：

```python
pip install　git+https://github.com/aleju/imgaug
```

方式二（安装pypi版本）：

```python
pip install imgaug
```

# Bounding Boxes实现

### 读取原影像bounding boxes坐标

读取xml文件并使用ElementTree对xml文件进行解析，找到每个object的坐标值。

```python
def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有object节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin,ymin,xmax,ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist
```

### 生成变换后的bounding boxe坐标文件

传入目标变换后的bounding boxe坐标，将原坐标替换成新坐标并生成新的xml文件。

```python
def change_xml_list_annotation(root, image_id, new_target,saveroot,id):

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))
```

### 生成变换序列

产生一个处理图片的Sequential。

```python
seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),
        # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])
```

### bounding box 变化后坐标计算

先读取该影像对应xml文件，获取所有目标的bounding boxes，然后依次计算每个box变化后的坐标。

```python
bndbox = read_xml_annotation(XML_DIR, name)
for epoch in range(AUGLOOP):
    seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

    # 读取图片
    img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
    img = np.array(img)

    # bndbox 坐标增强
    for i in range(len(bndbox)):
        bbs = ia.BoundingBoxesOnImage([
        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
        ], shape=img.shape)

        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        boxes_img_aug_list.append(bbs_aug)
```

# 使用示例

## 数据准备

输入数据为两个文件夹一个是需要增强的影像数据（JPEGImages），一个是对应的xml文件（Annotations）。**注意：影像文件名需和xml文件名相对应！**

![Annotations](https://img-blog.csdnimg.cn/20181125152842316.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)

![JPEGImages](https://img-blog.csdnimg.cn/20181125152918559.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)

## 设置文件路径

```python
    IMG_DIR = "./dataset/JPEGImages" #输入的影像文件夹路径
    XML_DIR = "./dataset/Annotations" # 输入的XML文件夹路径


    AUG_XML_DIR = "./dataset/AUG_XML" # 存储增强后的XML文件夹路径
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = "./dataset/AUG_IMG" # 存储增强后的影像文件夹路径
    mkdir(AUG_IMG_DIR)
```



## 设置增强次数

```python
    AUGLOOP = 10 # 每张影像增强的数量
```

## 设置增强参数

通过修改Sequential函数参数进行设置，具体设置参考[imgaug使用文档](https://imgaug.readthedocs.io/en/latest/index.html)

```python
seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 50% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 0.5)),
         # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  
    ])
```



## 输出

运行XMLaug.py ，运行结束后即可得到增强的影像和对应的xml文件夹
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181125153058242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181125153108215.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181125153413445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181125153425958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181125153433430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2018112515344467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Nvb29vMGw=,size_16,color_FFFFFF,t_70)
