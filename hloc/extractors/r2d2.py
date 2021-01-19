import sys
from pathlib import Path

from ..utils.base_model import BaseModel

r2d2_path = Path(__file__).parent / "../../third_party/r2d2"
sys.path.append(str(r2d2_path))
from extract import load_network, NonMaxSuppression, extract_multiscale
# from tools.dataloader import norm_RGB
import torchvision.transforms as tvf
import torch
RGB_mean = torch.cuda.FloatTensor([0.485, 0.456, 0.406])
RGB_std  = torch.cuda.FloatTensor([0.229, 0.224, 0.225])
norm_RGB = tvf.Normalize(mean=RGB_mean, std=RGB_std)

class R2D2(BaseModel):
    default_conf = {
        'checkpoint_name': 'r2d2_WASF_N16.pt',
        'top-k': 5000,

        'scale-f': 2**0.25,
        # TODO: set these parameters properly
        # range large enough to deal with all kinds of input?
        'min-size': 256, # 256,
        'max-size': 2400, # 1024,
        'min-scale': 0,
        'max-scale': 1,

        'reliability-thr': 0.7,
        'repetability-thr': 0.7,
    }
    # TODO: required_inputs is different from BaseModel's required_data_keys
    # so no check will be done in the forward function
    required_inputs = ['image']

    def _init(self, conf):
        model_fn = r2d2_path / "models" / conf['checkpoint_name']
        self.net = load_network(model_fn)
        # TODO: configure the parameters here
        self.detector = NonMaxSuppression(rel_thr=-1, rep_thr=-1)
        
    def _forward(self, data):
        # data is a dict containing 'name', 'image', and 'original_size'
        img = data['image']
        img = norm_RGB(img[0])[None]

        xys, desc, scores = extract_multiscale(self.net, img, self.detector,
                scale_f = self.conf['scale-f'],
                min_size = self.conf['min-size'],
                max_size = self.conf['max-size'],
                min_scale = self.conf['min-scale'],
                max_scale = self.conf['max-scale'],
        )
        idxs = scores.argsort()[-self.conf['top-k'] or None:]
        # xys = xys[idxs, :]
        # scale is stored as 32/s
        # s = 32 / xys[:, 2:]
        # xy = xys[:, :2] / s
        # x and y have been rescaled to match the original resolution!
        xy = xys[idxs, :2]
        desc = desc[idxs].t().contiguous()
        scores = scores[idxs]

        pred = {'keypoints': xy[None], 'descriptors': desc[None], 'scores': scores[None]}
        return pred
