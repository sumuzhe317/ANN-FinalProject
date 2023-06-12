import torch
from torchvision import datasets, transforms, models
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad, device, probability=0.5):
    if torch.rand(1) > probability:
        return image
    
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon*sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # normalize_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # perturbed_image = normalize_transforms(perturbed_image)
    # 返回对抗样本
    return perturbed_image

# def fgsm_attack(image, epsilon, data_grad, device, probability=0.5):
#     if image.size(0) > 1:
#         random_tensor = torch.rand(image.size(0))
#     else:
#         random_tensor = torch.tensor([torch.rand(1) for _ in range(image.size(0))])
#     random_mask = (random_tensor > probability).to(device)

    
#     # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
#     sign_data_grad = data_grad.sign()
#     sign_data_grad[random_mask] = 0
#     # 通过epsilon生成对抗样本
#     perturbed_image = image + epsilon*sign_data_grad
#     # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # normalize_transforms = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     # perturbed_image = normalize_transforms(perturbed_image)
#     # 返回对抗样本
#     return perturbed_image