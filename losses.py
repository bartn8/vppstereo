
import torch
import numpy as np

def sample_hints(hints, validhints, probability=0.20):
    new_validhints = (validhints * (torch.rand_like(validhints, dtype=torch.float32) < probability)).float()
    new_hints = hints * new_validhints  # zero invalid hints
    new_hints[new_validhints==0] = 0
    #new_hints[new_hints>5000] = 0
    return new_hints, new_validhints


def guided_metrics(disp, gt, valid):
    error = np.abs(disp-gt)
    error[valid==0] = 0
    
    bad1 = (error[valid>0] > 1.).astype(np.float32).mean()
    bad2 = (error[valid>0] > 2.).astype(np.float32).mean()
    bad3 = (error[valid>0] > 3.).astype(np.float32).mean()
    bad4 = (error[valid>0] > 4.).astype(np.float32).mean()
    avgerr = error[valid>0].mean()
    rms = (disp-gt)**2
    rms = np.sqrt( rms[valid>0].mean() )
    return {'bad 1.0':bad1, 'bad 2.0':bad2, 'bad 3.0': bad3, 'bad 4.0':bad4, 'avgerr':avgerr, 'rms':rms, 'errormap':error*(valid>0)}
