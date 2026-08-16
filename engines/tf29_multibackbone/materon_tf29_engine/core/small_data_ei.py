# -*- coding: utf-8 -*-
"""Identity-initialized low-rank E/I adapter for dense or RaggedTensor embeddings."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from .constraints import NonNegative
from .initializers import HarmonicInitializer
TensorLike = Union[tf.Tensor, tf.RaggedTensor]

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class SmallDataEIGate(tf.keras.layers.Layer):
    """Compact E/I residual adapter.

    The residual amplitude is ``alpha_max * (1-exp(-raw_alpha))`` with
    non-negative ``raw_alpha``. At raw_alpha=0 the layer is an exact identity.
    Training uses AdapterActivationCallback to set a tiny positive raw_alpha
    after the identity test, avoiding a dead adapter while keeping perturbation
    at training start negligible.
    """
    def __init__(self, feature_dim:int, context_dim:int, rank:int=4,
                 alpha_max:float=0.15, kernel_l2:float=1e-4,
                 alpha_penalty:float=1e-4, neutral_gate_penalty:float=1e-4,
                 use_layer_norm:bool=True, **kwargs:Any):
        super().__init__(**kwargs)
        self.feature_dim=int(feature_dim); self.context_dim=int(context_dim); self.rank=int(rank)
        self.alpha_max=float(alpha_max); self.kernel_l2=float(kernel_l2)
        self.alpha_penalty=float(alpha_penalty); self.neutral_gate_penalty=float(neutral_gate_penalty)
        self.use_layer_norm=bool(use_layer_norm)
        if self.feature_dim<=0 or self.rank<=0 or self.context_dim<0: raise ValueError("Invalid adapter dimensions")
        reg=tf.keras.regularizers.l2(self.kernel_l2) if self.kernel_l2>0 else None
        self.norm=tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="norm") if use_layer_norm else None
        self.down=tf.keras.layers.Dense(self.rank, activation=tf.nn.swish, kernel_initializer=HarmonicInitializer(0.06,0.2), bias_initializer="zeros", kernel_regularizer=reg, name="down")
        self.exc=tf.keras.layers.Dense(self.rank, activation=tf.nn.softplus, kernel_initializer=HarmonicInitializer(0.04,0.8), bias_initializer=tf.keras.initializers.Constant(-0.3), kernel_regularizer=reg, name="excitation")
        self.inh=tf.keras.layers.Dense(self.rank, activation=tf.nn.softplus, kernel_initializer=HarmonicInitializer(0.04,2.1), bias_initializer=tf.keras.initializers.Constant(-0.3), kernel_regularizer=reg, name="inhibition")
        self.comp=tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros", bias_initializer="zeros", kernel_regularizer=reg, name="competition")
        self.up=tf.keras.layers.Dense(self.feature_dim, use_bias=False, kernel_initializer=HarmonicInitializer(0.035,1.3), kernel_regularizer=reg, name="up")
        self.raw_alpha=self.add_weight("raw_alpha", shape=(), initializer="zeros", trainable=True, constraint=NonNegative(), dtype=tf.float32)
        self._analysis={}; self._symbolic_analysis={}
    @staticmethod
    def _is_ragged(x): return isinstance(x,tf.RaggedTensor) or (hasattr(x,"flat_values") and hasattr(x,"with_flat_values"))
    @classmethod
    def _flat(cls,x): return x.flat_values if cls._is_ragged(x) else x
    @classmethod
    def _restore(cls,ref,v): return ref.with_flat_values(v) if cls._is_ragged(ref) else v
    @property
    def alpha(self): return tf.cast(self.alpha_max, self.raw_alpha.dtype)*(1.0-tf.exp(-self.raw_alpha))
    def set_activation(self, raw_value:float=0.01): self.raw_alpha.assign(max(float(raw_value),0.0))
    def reset_identity(self): self.raw_alpha.assign(0.0)
    def compute_components(self, embeddings:TensorLike, context:Optional[TensorLike]=None, training:bool=False):
        x=tf.cast(self._flat(embeddings),tf.float32)
        tf.debugging.assert_equal(tf.shape(x)[-1],self.feature_dim)
        h=self.down(self.norm(x,training=training) if self.norm else x,training=training)
        e=self.exc(h,training=training); i=self.inh(h,training=training)
        if self.context_dim:
            if context is None: p=tf.zeros((tf.shape(x)[0],self.context_dim),tf.float32)
            else:
                p=tf.cast(self._flat(context),tf.float32)
                tf.debugging.assert_equal(tf.shape(p)[0],tf.shape(x)[0]); tf.debugging.assert_equal(tf.shape(p)[-1],self.context_dim)
            gate_in=tf.concat([h,p],axis=-1)
        else:
            p=tf.zeros((tf.shape(x)[0],0),tf.float32); gate_in=h
        g=self.comp(gate_in,training=training)
        gate_kernel=tf.cast(self.comp.kernel,tf.float32)
        gate_bias=tf.cast(self.comp.bias,tf.float32)
        gate_logit=tf.matmul(gate_in,gate_kernel)+gate_bias
        latent_contribution=h*tf.reshape(gate_kernel[:self.rank,0],(1,self.rank))
        context_contribution=p*tf.reshape(gate_kernel[self.rank:,0],(1,self.context_dim))
        signed=g*e-(1.0-g)*i
        delta=self.up(signed,training=training)
        a=tf.cast(self.alpha,x.dtype)
        y=x+a*delta
        if self.alpha_penalty>0: self.add_loss(self.alpha_penalty*tf.abs(a))
        if self.neutral_gate_penalty>0: self.add_loss(self.neutral_gate_penalty*tf.reduce_mean(tf.square(g-0.5)))
        ab=tf.zeros_like(g)+a
        analysis={"alpha":self._restore(embeddings,ab),"competition":self._restore(embeddings,g),"competition_logit":self._restore(embeddings,gate_logit),"latent_contribution":self._restore(embeddings,latent_contribution),"physical_context":self._restore(embeddings,p),"context_contribution":self._restore(embeddings,context_contribution),"excitation":self._restore(embeddings,e),"inhibition":self._restore(embeddings,i),"signed_latent":self._restore(embeddings,signed),"delta":self._restore(embeddings,delta)}
        self._analysis=analysis
        try:
            if any(tf.keras.backend.is_keras_tensor(self._flat(v)) for v in analysis.values()): self._symbolic_analysis=analysis
        except Exception: pass
        return self._restore(embeddings,y),analysis
    def call(self,inputs,training=False,return_analysis=False):
        if isinstance(inputs,(list,tuple)): x=inputs[0]; p=inputs[1] if len(inputs)>1 else None
        else: x=inputs; p=None
        y,a=self.compute_components(x,p,training=training)
        return (y,a) if return_analysis else y
    def get_analysis_tensors(self,symbolic=False): return dict(self._symbolic_analysis if symbolic else self._analysis)
    def get_config(self):
        c=super().get_config(); c.update({"feature_dim":self.feature_dim,"context_dim":self.context_dim,"rank":self.rank,"alpha_max":self.alpha_max,"kernel_l2":self.kernel_l2,"alpha_penalty":self.alpha_penalty,"neutral_gate_penalty":self.neutral_gate_penalty,"use_layer_norm":self.use_layer_norm}); return c

@tf.keras.utils.register_keras_serializable(package="MateronTF29")
class AdapterActivationCallback(tf.keras.callbacks.Callback):
    def __init__(self, raw_alpha:float=0.01, **kwargs): super().__init__(**kwargs); self.raw_alpha=float(raw_alpha)
    def on_train_begin(self, logs=None):
        del logs
        count=0
        for layer in self.model.submodules:
            # Supports both graph E/I gates and the composition-simplex E/I gate
            # without importing the latter here (avoids circular imports).
            if hasattr(layer,"set_activation") and hasattr(layer,"raw_alpha"):
                layer.set_activation(self.raw_alpha); count+=1
        if count==0: raise RuntimeError("Materon variant contains no activatable E/I adapter")
    def get_config(self): return {"raw_alpha":self.raw_alpha}
