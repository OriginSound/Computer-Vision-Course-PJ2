# Computer-Vision-Course-PJ2

## Classification 
We explored three kinds of augmentation methods. The accuracy of each model on cifar 100 is shown in the following table. **Passwords are all 1111**.

#### ResNet18
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Augmentation</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">ResNet18</td>
<td align="center">----</td>
<td align="center">76.5%</td>
<td align="center"><a href="https://pan.baidu.com/s/1K7y50PvgMlVAqfYFzoLskQ">
checkpoint</a></td>
</tr>

 <tr><td align="left">ResNet18</td>
<td align="center"> MixUp</td>
<td align="center">78.3%</td>
<td align="center"><a href="https://pan.baidu.com/s/1AzAlwf00U2d-vOELgnZ_oQ">checkpoint</a></td>
</tr>

 <tr><td align="left">ResNet18</td>
<td align="center"> CutOut</td>
<td align="center">77.3%</td>
<td align="center"><a href="https://pan.baidu.com/s/1PTb2piyKGUHDFw7bOf6BOA">checkpoint</a></td>
</tr>

 <tr><td align="left">ResNet18</td>
<td align="center"> CutMix</td>
<td align="center">79.6%</td>
<td align="center"><a href="https://pan.baidu.com/s/1tDl9KxyxXqredsYsdh-ihw">checkpoint</a></td>
</tr>

</tbody></table>

#### VGG16
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Augmentation</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">VGG16</td>
<td align="center">----</td>
<td align="center">73.8%</td>
<td align="center"><a href="https://pan.baidu.com/s/1msMpEcS_Qu3u8hoigz5MpA">
checkpoint</a></td>
</tr>

 <tr><td align="left">VGG16</td>
<td align="center"> MixUp</td>
<td align="center">72.8%</td>
<td align="center"><a href="https://pan.baidu.com/s/1sfcIRokhWM4fmY0QNLu0Pg">checkpoint</a></td>
</tr>

 <tr><td align="left">VGG16</td>
<td align="center"> CutOut</td>
<td align="center">73.9%</td>
<td align="center"><a href="https://pan.baidu.com/s/1AWfKnPtx4QsLGcaRmdn1eQ">checkpoint</a></td>
</tr>

 <tr><td align="left">VGG16</td>
<td align="center"> CutMix</td>
<td align="center">73.9%</td>
<td align="center"><a href="https://pan.baidu.com/s/1Af41tlx68tTPDrcf1gd5Lg">checkpoint</a></td>
</tr>

</tbody></table>

#### GoogleNet
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">Augmentation</th>
<th valign="bottom">Accuracy</th>
<th valign="bottom">checkpoint</th>

 <tr><td align="left">GoogleNet</td>
<td align="center">----</td>
<td align="center">76.5%</td>
<td align="center"><a href="https://pan.baidu.com/s/1tJaTZr2DwVX33xzixjpypg">
checkpoint</a></td>
</tr>

 <tr><td align="left">GoogleNet</td>
<td align="center"> MixUp</td>
<td align="center">78.3%</td>
<td align="center"><a href="https://pan.baidu.com/s/1hvTi4GtDkt38JRVU8F_pVQ">checkpoint</a></td>
</tr>

 <tr><td align="left">GoogleNet</td>
<td align="center"> CutOut</td>
<td align="center">77.3%</td>
<td align="center"><a href="https://pan.baidu.com/s/13uGw4tTcGSXlbIKaRfZXKw">checkpoint</a></td>
</tr>

 <tr><td align="left">GoogleNet</td>
<td align="center"> CutMix</td>
<td align="center">79.6%</td>
<td align="center"><a href="https://pan.baidu.com/s/1WYbCHJjHs4BHl2_uGt7D1Q">checkpoint</a></td>
</tr>

</tbody></table>

## Detection 

The config files for faster R-CNN, FCOS, and YOLOv3 are shown in the following table.
|   Model         | config name  | Download |
|:---------------:|:-----------:|:---------:|
| Faster R-CNN  | [Faster R-RNN](https://github.com/OriginSound/Computer-Vision-Course-PJ2/blob/main/detection/configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py) | [model](https://pan.baidu.com/s/1xSORdAJLN0k3O7GpfLSZKQ)  |
|FCOS | [FCOS](https://github.com/OriginSound/Computer-Vision-Course-PJ2/blob/main/detection/configs/pascal_voc/fcos_4x4.py) | [model](https://pan.baidu.com/s/1xSORdAJLN0k3O7GpfLSZKQ)  |
|YOLOv3 | [YOLOv3](https://github.com/OriginSound/Computer-Vision-Course-PJ2/blob/main/detection/configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py) | [model](https://pan.baidu.com/s/1xSORdAJLN0k3O7GpfLSZKQ)  |