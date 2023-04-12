import torch
import new_proton
import copy
import freprecess

def flow_vassel(data, flow_time, info, new_prot=None):
    if new_prot==None:
        return
    
    center_index = (info.fov / info.delta) // 2
    lower, upper = int(center_index - info.bandwidth // 2), int(center_index + info.bandwidth // 2)
    vassel = copy.deepcopy(data[:, lower:upper, :])
    vassel = torch.roll(vassel, 1, 0)
    vassel[0, :, :, :] = new_prot.output(new_prot.tensor, N=flow_time)
    # print(data[:, lower:upper, 0])
    print(vassel[0, :, 0, 1])
    print(data[-1, lower:upper, 0, 1])
    flow_time += 1
    data[:, lower:upper, :, :] = vassel
    return flow_time, data

def freeprecess(data, time, index_list, flow_time,
                gradient:list, new_prot=None, frame_prot=None, gradient_mat=None, info=None, flow=False):
    
    if len(gradient) == 0:
        A, B = freprecess.res(time, info.T1[0], info.T2[0], 0 + info.w0)
        for (i, j) in index_list[0]:
            data[i, j, :] = data[i, j, :] @ A.T + B
        # 对新质子流入的操作
        if flow == False: # 没有gradient
            if new_prot != None:
                new_prot.record(t=time, n=flow_time)
            frame_prot.record(t=time, n=flow_time) # 其它量都不会操作
        
        A, B = freprecess.res(time, info.T1[1], info.T2[1], 0 + info.w0)
        for (i, j) in index_list[1]:
            data[i, j, :] = data[i, j, :] @ A.T + B

    else:
        assert flow == True
        for (i, j) in index_list[0]:
            df = gradient_mat[i, j]
            A, B = freprecess.res(time, info.T1[0], info.T2[0], df + info.w0)
            data[i, j, :] = data[i, j, :] @ A.T + B
        # 对新质子流入的操作
        if len(gradient) == 1: # Gx
            new_prot.record(t=time, Gx=gradient[0], n=flow_time)
        elif len(gradient) == 2: # Gxy
            new_prot.record(t=time, Gx=gradient[0], Gy=gradient[1], n=flow_time)

        for (i, j) in index_list[1]:
            df = gradient_mat[i, j]
            A, B = freprecess.res(time, info.T1[1], info.T2[1], df + info.w0)
            data[i, j, :] = data[i, j, :] @ A.T + B
    return data, new_prot

def free_flow(data, time, time_before, flow_time, info, new_prot=None, 
              frame_prot=None, grad=[], etf=None, flow=False,
              gradient_mat=None, index_list=None): # 用于 tau_y 或者 TR
    # gradient 统一乘以gamma过了, 取值: False, [Gx], [Gx, Gy]
    # time_before 表示之前的 self.time
    # etf: each_time_flow
    # (new_prot == None) 就不考虑 flow
    # assert new_prot != None
    new = 0
    flow_num = int((time_before + time) // etf)
    # before_time = self.etf - self.time
    if (time_before + time) % etf == (time_before + time):
        rest_time = time
        time_before += time
    else:
        rest_time = (time_before + time) % etf
        new_time_before = rest_time
        new = 1

    if flow == False: # new = 1, 这里针对 readout 外的 relax
        data, _ = freeprecess(data, time=time, gradient=grad, new_prot=new_prot, frame_prot=frame_prot, flow=False,
                           gradient_mat=gradient_mat, info=info, flow_time=flow_time, index_list=index_list)
        if new == 1:
            return data, new_time_before # new_prot == None, 用 _ 接收
        else:
            return data, time_before
    
    # 满足flow条件时 进行flow
    assert new_prot != None
    for n in range(flow_num):
        t = etf - time_before if n == 0 else etf
        data, new_prot = freeprecess(data, time=t, gradient=grad, new_prot=new_prot,flow=True,
                           gradient_mat=gradient_mat, info=info, flow_time=flow_time, index_list=index_list)
        # 将当前时刻下做好的 new proton 提供出来
        flow_time, data = flow_vassel(data, flow_time, info, new_prot)
    data, new_prot = freeprecess(data, time=rest_time, gradient=grad, new_prot=new_prot,flow=True,
                           gradient_mat=gradient_mat, info=info, flow_time=flow_time, index_list=index_list)
    if new == 1:
        return data, new_time_before, new_prot, flow_time
    else:
        return data, time_before, new_prot, flow_time