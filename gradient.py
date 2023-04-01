import torch
import image

class transverse_G:
    def __init__(self, ksapce, info) -> None:
        self.kspace_info = ksapce
        # self.body = body
        self.fov = info.fov
        self.B0 = info.b0
        self.gamma = info.gamma
        self.delta = self.kspace_info.delta # spatial resolution 这里是cm
        # w0 目前暂时不要直接加入计算中
        

    def get_Gx(self):
        # Gx_tensor = torch.arange(-self.fov/2,self.fov/2,self.delta) * self.Gx
        return self.Gx
    
    def get_Gy(self, num_rf:int):
        # G_diff = self.Gyp - num_rf * self.Gyi
        # Gy_tensor = torch.arange(-self.fov/2,self.fov/2,self.delta) * G_diff
        return self.Gyp, self.Gyi