import cv2
import numpy as np
from PIL import Image
import os

#第二截取特定矩形框的图像并放大2倍保存在原图像的右下方
# 遍历指定目录，显示目录下的所有文件名
def CropImage4File(filepath,destpath):
    pathDir =  os.listdir(filepath)    # 列出文件路径中的所有路径或文件
    for allDir in pathDir:
        a, b = os.path.splitext(allDir)#分离文件名和扩展名
        child = os.path.join(filepath, allDir)#连接路径
        if os.path.isfile(child):
            image = cv2.imread(child)
            h,w=image.shape[:2]
            x1 = 130
            y1 = 130
            x2 = 190
            y2 = 180
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(destpath +a+b, image)
            cropImage = image[y1:y2, x1:x2]  #裁剪图像
            height,width = cropImage.shape[:2]  #获取裁剪图像的高度和宽度。
            enhanced_image = cv2.resize(cropImage,(3*width,3*height),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(croppath+a+b,enhanced_image)

def cover(IMAGE_PATH1,IMAGE_PATH2,IMAGE_SAVE_PATH):
    IMAGE_FORAMT={'.jpg','.png'}
    #获取文件中图片的数目
    image_names1=[name for name in os.listdir(IMAGE_PATH1)]
    image_names1.sort()
    image_names2=[name for name in os.listdir(IMAGE_PATH2)]
    image_names2.sort()
    image_number=len(image_names1)

    for i in range(image_number):
 
        image=Image.open(os.path.join(IMAGE_PATH1+'\\'+image_names1[i]))
        #image.show()
        w=image.size[0]
        h=image.size[1]
   
        toImage = Image.new('RGB', (w, h))
        toImage.paste(image,(0,0))#图像裁剪覆盖
        
        enhanced_image=Image.open(os.path.join(IMAGE_PATH2+'\\'+image_names2[i]))
        width=enhanced_image.size[0]
        height=enhanced_image.size[1]


        toImage.paste(enhanced_image,(w-width,h-height,w,h))

        toImage.save(IMAGE_SAVE_PATH+image_names1[i])

if __name__ == '__main__':
    filepath ='./rect/original/100H/21/' #源图像
    destpath='./rect/resize/100H/21/' # resized images saved here 存放在原图中画框的图像
    croppath='./rect/crop/100H/21/'#存放裁剪部分
    IMAGE_SAVE_PATH='./rect/cat/100H/21/'#存放拼接的地址
    CropImage4File(filepath,destpath,)
    cover(destpath,croppath,IMAGE_SAVE_PATH)

    # img_path = './result/rect_crop/original/'
    # paths = os.listdir(img_path)
    # for i in range(24):
    #     img = cv2.imread(img_path+paths[i])
    #     print(img)
    #     img1 = cv2.resize(img,dsize=(480,320))
    #     # img1 = cv2.resize(img,dsize=(480,320))
    #     cv2.imwrite(img_path+paths[i],img1)

    # img = cv2.imread(img_path)
    # img1 = cv2.resize(img,dsize=(480,320))
    # cv2.imwrite(img_path,img1)