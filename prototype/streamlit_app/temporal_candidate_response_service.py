from __future__ import annotations
from functools import lru_cache
import importlib.util, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'models/candidate_dose_response/candidate_response_v7_ensemble_20260710/temporal_v5_oocyte_mii.pt'
V5=ROOT/'scripts/experiment/phase870_candidate_dose_response/train_stage_aware_temporal_candidate_response.py'
@lru_cache(maxsize=1)
def _bundle():
 if not PACKAGE.exists(): raise FileNotFoundError(PACKAGE)
 return torch.load(PACKAGE,map_location='cpu',weights_only=False)
@lru_cache(maxsize=1)
def _net_class():
 sp=importlib.util.spec_from_file_location('v5_temporal_inference',V5);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m.Net
def available():
 try:_bundle();return True,''
 except Exception as e:return False,str(e)
@lru_cache(maxsize=1)
def _models():
 b=_bundle();Net=_net_class();hp=b['trial'];out={}
 for task,item in b['models'].items():
  net=Net(len(b['static_features']),len(b['dynamic_features']),int(hp['hidden']),float(hp['dropout']),False)
  net.load_state_dict(item['state_dict']);net.eval();out[task]=net
 return out
def predict(candidate_row:Mapping[str,Any],history_snapshots:list[Mapping[str,Any]]|None):
 b=_bundle();hist=[]
 for item in history_snapshots or []:
  if isinstance(item,pd.Series):hist.append(item.to_dict())
  elif isinstance(item,Mapping):hist.append(dict(item))
 hist=hist or [dict(candidate_row)]
 sf=list(b['static_features']);df=list(b['dynamic_features']);static=pd.DataFrame([candidate_row]).reindex(columns=sf)
 x=b['static_scaler'].transform(b['static_imputer'].transform(static)).astype('float32')
 h=pd.DataFrame(hist).reindex(columns=df).tail(int(b['max_visits']))
 z=b['dynamic_scaler'].transform(b['dynamic_imputer'].transform(h)).astype('float32')
 seq=np.zeros((1,int(b['max_visits']),len(df)),dtype='float32');mask=np.zeros((1,int(b['max_visits'])),dtype='float32');seq[0,:len(z)]=z;mask[0,:len(z)]=1
 day=float(candidate_row.get('gn_day',candidate_row.get('Day',1)) or 1);stage=np.array([0 if day<=4 else 1 if day<=7 else 2],dtype='int64')
 out={}
 for task,net in _models().items():
  with torch.no_grad(): y=net(torch.tensor(x),torch.tensor(seq),torch.tensor(mask),torch.tensor(stage))
  out[task]=float(y.numpy()[0])
 return out
