"""Independent numerical oracle: replay every recorded operator with CPU PyTorch float64."""
import json
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
with open('tmp/workstation-fixtures.json', encoding='utf-8') as stream:
    fixtures=json.load(stream)
maximum=0.; count=0
for fixture in fixtures:
    data={t['id']:torch.tensor(t['values'],dtype=torch.float64).reshape(t['shape']) for t in fixture['tensors']}
    for step in fixture['steps']:
        xs=[data[key] for key in step['inputs']]; x=xs[0]; kind=step['kind']; s=step['settings']; expected=data[step['output']]
        if kind=='Linear': y=F.linear(x,xs[1],xs[2] if len(xs)>2 else None)
        elif kind in ('Conv1d','Conv2d','Conv3d'): y=getattr(F,kind.lower())(x,xs[1],xs[2] if len(xs)>2 else None,stride=s['stride'],padding=s['padding'],dilation=s['dilation'],groups=s['groups'])
        elif kind.startswith('ConvTranspose'): y=getattr(F,'conv_transpose'+kind[-2:].lower())(x,xs[1],xs[2],stride=s['stride'],padding=s['padding'],output_padding=s['outputPadding'],groups=s['groups'],dilation=s['dilation'])
        elif kind=='Upsample': y=F.interpolate(x,scale_factor=s['scale'],mode=s['mode'],align_corners=s['alignCorners'] if s['mode']=='bilinear' else None)
        elif kind=='BatchNorm2d':
            y=F.batch_norm(x,None if s['training'] else torch.zeros(x.shape[1],dtype=x.dtype),None if s['training'] else torch.ones(x.shape[1],dtype=x.dtype),xs[1],xs[2],s['training'],eps=s['eps'])
        elif kind=='InstanceNorm2d': y=F.instance_norm(x,weight=xs[1],bias=xs[2],eps=s['eps'])
        elif kind=='GroupNorm': y=F.group_norm(x,s['groups'],xs[1],xs[2],s['eps'])
        elif kind=='LayerNorm': y=F.layer_norm(x,[x.shape[-1]],xs[1],xs[2],s['eps'])
        elif kind=='RMSNorm': y=F.rms_norm(x,[x.shape[-1]],xs[1],s['eps'])
        elif kind=='Flatten': y=x.reshape(s['shape'])
        elif kind=='Softmax': y=x.softmax(-1)
        elif kind=='MaxPool2d': y=F.max_pool2d(x,s['k'],s['stride'])
        elif kind=='AvgPool2d': y=F.avg_pool2d(x,s['k'],s['stride'])
        elif kind=='AdaptiveAvgPool2d': y=F.adaptive_avg_pool2d(x,1)
        elif kind=='Dropout': y=x*xs[1]/(1-s['p'])
        elif kind=='Identity':
            if s.get('operation') in ('norm-mean','norm-var'):
                mode=s['kind']; var=s['operation']=='norm-var'
                if mode in ('LayerNorm','RMSNorm'): z=x.reshape(-1,x.shape[-1])
                elif mode=='BatchNorm2d': z=x.movedim(1,0).reshape(x.shape[1],-1)
                elif mode=='GroupNorm': z=x.reshape(x.shape[0]*s['groups'],-1)
                else: z=x.reshape(x.shape[0]*x.shape[1],-1)
                if mode=='BatchNorm2d' and not s['training']: y=torch.full((z.shape[0],),1. if var else 0.,dtype=x.dtype)
                elif mode=='RMSNorm': y=(z*z).mean(-1) if var else torch.zeros(z.shape[0],dtype=x.dtype)
                else: y=z.var(-1,unbiased=False) if var else z.mean(-1)
            elif s.get('operation')=='row-max': y=x.amax(-1,keepdim=True)
            elif s.get('operation')=='row-exp': y=(x-xs[1]).exp()
            elif s.get('operation')=='row-sum': y=x.sum(-1,keepdim=True)
            elif s.get('operation')=='add': y=x+xs[1]
            elif s.get('operation')=='mul': y=x*xs[1]
            elif s.get('operation')=='one-minus': y=1-x
            elif s.get('operation')=='causal': y=x.masked_fill(torch.ones(x.shape[-2:],dtype=torch.bool).triu(1),-1e30)
            elif 'axes' in s: y=x.permute(s['axes'])
            elif 'start' in s: y=x.narrow(s['axis'],s['start'],s['end']-s['start'])
            elif 'axis' in s: y=torch.cat(xs,dim=s['axis'])
            elif 'transposeB' in s: y=(x@(xs[1].transpose(-2,-1) if s['transposeB'] else xs[1]))*s['scale']
            else: raise AssertionError(('Uncovered operation',step))
        elif kind=='LeakyReLU': y=F.leaky_relu(x,.2)
        elif kind=='Sigmoid': y=x.sigmoid()
        elif kind=='Tanh': y=x.tanh()
        else: y=getattr(F,kind.lower())(x)
        error=(y-expected).abs().max().item();maximum=max(maximum,error);count+=1
        assert torch.allclose(y,expected,atol=3e-7,rtol=3e-7), (fixture['id'],step['id'],kind,error)
print(f'PyTorch {torch.__version__}: {count} complete operator outputs matched across {len(fixtures)} configurations. Max absolute difference: {maximum:.3g} (GELU erf approximation allowed 3e-7).')
