import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def resize_with_padding(batch: torch.Tensor, h: int, w: int) -> torch.Tensor:
	d_ar = float(batch.shape[3]) / batch.shape[2]
	t_ar = float(w) / h
	if (t_ar > d_ar):
		scale_h = h
		scale_w = int(h * d_ar)
		pad_l = (w - scale_w) // 2
		pad_r = w - pad_l - scale_w
		pad_t = 0
		pad_b = 0
	else:
		scale_h = int(w / d_ar)
		scale_w = w
		pad_l = 0
		pad_r = 0
		pad_t = (h - scale_h) // 2
		pad_b = h - pad_t - scale_h
	scaled = F.interpolate(batch, (scale_h, scale_w))
	return F.pad(scaled, (pad_l, pad_r, pad_t, pad_b))
	
# with Image.open("s.jpg") as img:
# 	batch = transforms.ToTensor()(img).unsqueeze(0)
# 	out = resize_with_padding(batch, 100, 100)
# 	# out = transforms.ToTensor()(out)
# 	print(out.shape)
# 	transforms.ToPILImage()(out.squeeze(0)).show()
