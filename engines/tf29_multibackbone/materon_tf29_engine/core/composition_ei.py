"""Composition ei utilities for the TF2.9 multi-backbone engine."""

from __future__ import annotations
from typing import Any
import numpy as np
import tensorflow as tf
from .constraints import NonNegative
from .initializers import HarmonicInitializer

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class CompositionSimplexEIGate(tf.keras.layers.Layer):
    def __init__(self, prior_table, feature_names=None, rank=4, alpha_max=0.10, **kwargs:Any):
        super().__init__(**kwargs); p=np.asarray(prior_table,np.float32)
        if p.ndim!=2 or p.shape[0]<2: raise ValueError("prior_table shape invalid")
        self.prior=p[1:]; self.prior_list=self.prior.tolist(); self.n_elements=self.prior.shape[0]; self.context_dim=self.prior.shape[1]; self.feature_names=list(feature_names or []); self.rank=int(rank); self.alpha_max=float(alpha_max)
        self.element_proj=tf.keras.layers.Dense(rank,activation=tf.nn.swish,kernel_initializer=HarmonicInitializer(0.06,0.4),name="element_proj")
        self.global_proj=tf.keras.layers.Dense(rank,activation=tf.nn.swish,kernel_initializer=HarmonicInitializer(0.06,1.4),name="global_proj")
        self.exc=tf.keras.layers.Dense(1,activation=tf.nn.softplus,kernel_initializer=HarmonicInitializer(0.04,0.7),bias_initializer=tf.keras.initializers.Constant(-0.3),name="excitation")
        self.inh=tf.keras.layers.Dense(1,activation=tf.nn.softplus,kernel_initializer=HarmonicInitializer(0.04,2.0),bias_initializer=tf.keras.initializers.Constant(-0.3),name="inhibition")
        self.comp=tf.keras.layers.Dense(1,activation="sigmoid",kernel_initializer="zeros",bias_initializer="zeros",name="competition")
        self.raw_alpha=self.add_weight("raw_alpha",shape=(),initializer="zeros",trainable=True,constraint=NonNegative())
        self._analysis={}; self._symbolic_analysis={}
    @property
    def alpha(self): return self.alpha_max*(1.0-tf.exp(-self.raw_alpha))
    def set_activation(self,raw_value=0.01): self.raw_alpha.assign(max(float(raw_value),0.0))
    def reset_identity(self): self.raw_alpha.assign(0.0)
    @property
    def adapter_only_weights(self): return list(self.weights)
    def call(self,c,training=False):
        c=tf.cast(c,tf.float32)
        tf.debugging.assert_equal(tf.shape(c)[-1],self.n_elements)
        tf.debugging.assert_greater_equal(c,0.0)
        total=tf.reduce_sum(c,axis=-1,keepdims=True)
        tf.debugging.assert_positive(total)
        denominator=tf.maximum(total,1e-12)

        p=tf.constant(self.prior,tf.float32)
        global_ctx=tf.matmul(c,p)
        el=self.element_proj(p,training=training)
        gl=self.global_proj(global_ctx,training=training)[:,None,:]
        h=tf.nn.swish(el[None,:,:]+gl)
        e=self.exc(h,training=training)
        i=self.inh(h,training=training)
        g=self.comp(h,training=training)
        signed=tf.squeeze(g*e-(1-g)*i,-1)

        # Multiplicative simplex reweighting.
        logits=tf.clip_by_value(self.alpha*signed,-20.0,20.0)
        weighted=c*tf.exp(logits)
        transformed=weighted/tf.maximum(
            tf.reduce_sum(weighted,axis=-1,keepdims=True),1e-12
        )

        # Exact identity correction:
        # at alpha=0, weighted == c and transformed == base_normalized,
        # therefore the correction is exactly zero and y == c.  The
        # transformed branch remains differentiable with respect to raw_alpha,
        # so the gate can learn away from the identity initialization.
        base_normalized=c/denominator
        y=c+(transformed-base_normalized)

        self._analysis={
            "alpha":tf.zeros_like(c[:,:1])+self.alpha,
            "competition":tf.squeeze(g,-1),
            "excitation":tf.squeeze(e,-1),
            "inhibition":tf.squeeze(i,-1),
            "signed_score":signed,
            "physical_context":tf.broadcast_to(
                p[None,:,:],
                [tf.shape(c)[0],self.n_elements,self.context_dim]
            ),
            "simplex_input_sum":total,
            "simplex_output_sum":tf.reduce_sum(
                y,axis=-1,keepdims=True
            ),
        }
        try:
            if tf.keras.backend.is_keras_tensor(c):
                self._symbolic_analysis=self._analysis
        except Exception:
            pass
        return y
    def get_analysis_tensors(self,symbolic=False): return dict(self._symbolic_analysis if symbolic else self._analysis)
    def get_config(self):
        c=super().get_config(); c.update({"prior_table":[[0.0]*self.context_dim]+self.prior_list,"feature_names":self.feature_names,"rank":self.rank,"alpha_max":self.alpha_max}); return c
