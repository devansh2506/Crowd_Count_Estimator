# Crowd-Count-Estimation

Crowd counting – estimating the number of people in an image – is a key task in computer vision with applications in public safety, surveillance, and event management. It is challenging due to scale variations, heavy occlusion, perspective distortion, and background clutter. Traditional CNN methods (e.g. CSRNet) achieve strong results by regressing density maps. In this work, we implement a deep model on the ShanghaiTech dataset: **VUI-CrowdNet**, a CNN with a VGG-16 encoder, U-Net–style decoder, and an Inverse Attention Block (IAB) to suppress background. 

Our goal is to achieve high precision and segmentation awareness for accurate crowd counting.

## VUI-CrowdNet Architecture

We build VUI-CrowdNet on a VGG-16 encoder (pretrained on ImageNet). An input image is fed through VGG-16 up to the fifth pooling layer, producing feature maps that encode both low-level textures and high-level semantics. 

To restore full spatial resolution, we attach a U-Net–style decoder consisting of five upsampling blocks. Each block performs a 3×3 transposed convolution (stride 2) followed by a 3×3 convolution (both with 64 filters), effectively doubling the width and height of the feature maps at each stage. This careful upsampling preserves spatial detail, allowing the network to output a high-resolution density map. Maintaining original resolution avoids any loss of detail that resizing might cause.

After the final upsampling, we apply our **Inverse Attention Block (IAB)**. The IAB is a small 3-layer convolutional module that predicts an inverse attention mask A⁻¹ highlighting background regions. Formally, if F is the upsampled feature map (after the decoder), the IAB outputs an element-wise mask A⁻¹ of the same spatial size. We then compute a refined map:

F′ = F − F ⊙ A⁻¹

where ⊙ denotes elementwise multiplication. Intuitively, this subtracts out features associated with background, forcing the network to focus on crowd areas. This inversion scheme is key to making the counting easier by dimming non-crowd regions. Finally, we pass F′ through a 1x1 Convolution to produce the predicted density map.

![VUI-CrowdNet Architecture](https://github.com/user-attachments/assets/0ae8019e-0120-4c52-98d6-f02a3db078b6)

## Results

On the ShanghaiTech dataset, our best VUI-CrowdNet model achieved an MAE of ≈16.7. This is a competitive result: it improves upon many classical methods (e.g., Switching-CNN’s MAE 21.6) and demonstrates the value of segmentation guidance and U-Net decoding. 

The predicted density maps were qualitatively convincing – background areas were dimmed by the IAB and crowd blobs were highlighted in the correct locations. In test images, the network clearly focused on clusters of pedestrians, validating the approach of background suppression. The CNN yields spatially coherent density maps, and achieves lower error and faster training. 

In summary, injecting segmentation awareness into a CNN is an effective strategy for crowd counting.

## Video Demonstration

![VUI Crowd Net Video Demonstration](./VUI%20Crowd%20Net.mp4)

