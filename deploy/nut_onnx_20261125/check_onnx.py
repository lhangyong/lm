# -*- coding: utf-8 -*-
"""包自检：ONNX 形状 + 记忆库点数 + 阈值"""
import os, sys, json
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'model', 'metal_nut_patchcore_backbone.onnx')
NPZ = os.path.join(HERE, 'model', 'memory_bank.npz')
META = os.path.join(HERE, 'model', 'memory_meta.txt')

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
print('输入:', [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print('输出:', [(o.name, o.shape, o.type) for o in sess.get_outputs()])
z = np.load(NPZ)
print('记忆库:', z['features'].shape, z['features'].dtype, '| grid', int(z['grid_size']))
print('阈值:'); print(open(META, encoding='utf-8').read().strip())
out = sess.run(None, {'input': np.zeros((1, 3, 256, 256), dtype=np.float32)})[0]
print('零输入前向输出:', out.shape, '有限值' if np.isfinite(out).all() else '含 NaN/Inf')
