"""Schedule utilities for the TF2.9 multi-backbone engine."""

from kgcnn.training.scheduler import LinearLearningRateScheduler

def make_linear_callback(start,stop,epochs,decay_start=100): return LinearLearningRateScheduler(learning_rate_start=start,learning_rate_stop=stop,epo_min=decay_start,epo=epochs,verbose=0)
