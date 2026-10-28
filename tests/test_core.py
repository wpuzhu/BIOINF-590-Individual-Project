import unittest
import numpy as np
import torch
from kan_mlp.models import KANLayer,MODELS,architecture,build,parameter_count
from kan_mlp.data import synthetic,biological
from kan_mlp.study import fit

class CoreTests(unittest.TestCase):
 def test_budgets(self):
  for inputs,outputs in [(2,1),(8,10),(30,2)]:
   for depth in [1,2,3]:
    for kind in MODELS:
     budget=1024 if inputs==2 else 4096
     dims,p=architecture(kind,inputs,outputs,depth,budget)
     self.assertEqual(p,sum(q.numel() for q in build(kind,dims).parameters()))
     self.assertLessEqual(abs(p-budget)/budget,0.05)
 def test_large_shallow_budget(self):
  for kind in MODELS:
   dims,p=architecture(kind,2,1,1,4096)
   self.assertLessEqual(abs(p-4096)/4096,.05)
   self.assertEqual(p,sum(q.numel() for q in build(kind,dims).parameters()))
 def test_gradients(self):
  for family in ['spline','fourier','rbf']:
   layer=KANLayer(2,2,family).double()
   x=torch.tensor([[.123,-.456],[.314,.271]],dtype=torch.float64,requires_grad=True)
   self.assertTrue(torch.autograd.gradcheck(layer,(x,),eps=1e-6,atol=1e-4))
 def test_splits(self):
  d=synthetic('waves',9)
  self.assertEqual(len(set(d['split_ids'].values())),4)
  self.assertTrue((np.abs(d['inside'][0])<=1).all())
  self.assertTrue((np.abs(d['outside'][0])>1).any(1).all())
  self.assertTrue(np.array_equal(d['val'][0],synthetic('waves',9,n=200)['val'][0]))
 def test_real_split_and_scaler(self):
  d=biological('breast_cancer',4)
  ids=d['split_ids']
  for a,b in [('train','val'),('train','inside'),('val','inside')]:
   self.assertFalse(set(ids[a])&set(ids[b]))
  np.testing.assert_allclose(d['train'][0].mean(0),0,atol=1e-6)
 def test_best_checkpoint(self):
  cfg=dict(terms=8,weight_decay=0,max_epochs=15,check_every=1,lr_patience=2,min_delta=.001,patience=5)
  d=synthetic('linear',4,n=50)
  model,r=fit('mlp-silu',[2,8,1],d,cfg,1,.01)
  with torch.no_grad():
   x,y,_=d['val'];v=float((model(torch.from_numpy(x))-torch.from_numpy(y)).square().mean())
  self.assertAlmostEqual(v,r['val_loss'],places=6)

 def test_interactions_need_composition(self):
  torch.manual_seed(12)
  x=torch.tensor([[.13,.27]],dtype=torch.float64,requires_grad=True)
  single=KANLayer(2,1,'rbf').double()
  first=torch.autograd.grad(single(x).sum(),x,create_graph=True)[0][0,0]
  mixed=torch.autograd.grad(first,x)[0][0,1]
  self.assertAlmostEqual(float(mixed.detach()),0,places=12)
  composed=build('kan-rbf',[2,3,1]).double()
  first=torch.autograd.grad(composed(x).sum(),x,create_graph=True)[0][0,0]
  mixed=torch.autograd.grad(first,x)[0][0,1]
  self.assertGreater(abs(float(mixed.detach())),1e-9)

if __name__=='__main__':unittest.main()
