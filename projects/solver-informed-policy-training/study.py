"""Solver-reward policy gradients over a finite formulation DSL; tiny LM smoke backend."""
from __future__ import annotations
import argparse
import copy
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import milp,Bounds,LinearConstraint
CANDIDATES=[' '.join(x) for x in itertools.product(('min','max'),('le','ge'),('real','int'))]


def tasks(seed,n=24,shift=False):
    g=np.random.default_rng(seed);out=[]
    for i in range(n):
        target=CANDIDATES[i%8];sense,relation,kind=target.split();rhs=round(float(g.uniform(1,5)*(3 if shift else 1)),3)
        upper=round(2*rhs+4,3)
        text=f"{sense}imize x; x {'at most' if relation=='le' else 'at least'} {rhs:.3f}; x {'integer' if kind=='int' else 'real'}; 0<=x<={upper:.3f}."
        out.append({'prompt':text,'target':target,'rhs':rhs,'upper':upper})
    return out


def solve(task,model):
    if model not in CANDIDATES:raise ValueError('Candidate must belong to restricted DSL')
    sense,rel,kind=model.split();rhs,upper=task['rhs'],task['upper']
    if not np.isfinite([rhs,upper]).all() or not 0<rhs<upper:raise ValueError('Invalid task bounds')
    r=milp([1. if sense=='min' else -1.],integrality=[int(kind=='int')],bounds=Bounds([0],[upper]),
           constraints=LinearConstraint([[1]],[-np.inf if rel=='le' else rhs],[rhs if rel=='le' else np.inf]),
           options={'time_limit':2.,'mip_rel_gap':0.0})
    if r.status!=0 or r.x is None:return None
    return float(r.x[0])


def feasible(task,model,x):
    _,rel,kind=model.split()
    return (0<=x<=task['upper'] and (x<=task['rhs'] if rel=='le' else x>=task['rhs'])
            and (kind=='real' or abs(x-round(x))<1e-8))


def reward(task,candidate):
    """Exact solves + known synthetic semantics. No LLM judge and no arbitrary execution."""
    if candidate not in CANDIDATES:return 0.0
    probes=[0.,.5,task['rhs']-.1,task['rhs']+.1,task['upper'],round(task['rhs']),task['upper']-.25]
    if any(feasible(task,candidate,x)!=feasible(task,task['target'],x) for x in probes):return 0.0
    x=solve(task,candidate);truth=solve(task,task['target'])
    return float(x is not None and truth is not None and abs(x-truth)<1e-6)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(128,16);self.rnn=nn.GRU(16,32,batch_first=True);self.head=nn.Linear(32,128)
    def forward(self,ids,attention_mask=None):
        hidden,_=self.rnn(self.embedding(ids));return self.head(hidden)


def ascii_encode(text):
    ids=[ord(c) for c in text]
    if any(i>=128 for i in ids):raise ValueError('Tiny backend accepts ASCII only')
    return ids


def scores(model,batch,encode=ascii_encode,pad_id=0):
    seq=[];starts=[]
    for task in batch:
        prefix=encode(task['prompt']+'\nModel: ')
        for c in CANDIDATES:seq.append(prefix+encode(c+'\n'));starts.append(len(prefix))
    maxlen=max(map(len,seq));ids=torch.full((len(seq),maxlen),pad_id,dtype=torch.long)
    attention=torch.zeros_like(ids);mask=torch.zeros((len(seq),maxlen-1))
    for i,t in enumerate(seq):
        ids[i,:len(t)]=torch.tensor(t);attention[i,:len(t)]=1;mask[i,starts[i]-1:len(t)-1]=1
    device=next(model.parameters()).device;ids=ids.to(device);attention=attention.to(device);mask=mask.to(device)
    result=model(ids,attention_mask=attention);logits=result if isinstance(result,torch.Tensor) else result.logits
    logp=logits[:,:-1].log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    return (logp*mask).sum(1).reshape(len(batch),len(CANDIDATES))


def train(model,batch,mode='rl',steps=60,seed=0,encode=ascii_encode,pad_id=0):
    if mode not in ('sft','rl') or not batch or steps<1:raise ValueError('Invalid training settings')
    torch.manual_seed(seed);model.train();o=torch.optim.Adam(model.parameters(),lr=.004)
    with torch.no_grad():reference=scores(model,batch,encode,pad_id).log_softmax(-1).detach()
    device=reference.device
    rewards=torch.tensor([[reward(t,c) for c in CANDIDATES] for t in batch],device=device)
    target=torch.tensor([CANDIDATES.index(t['target']) for t in batch],device=device)
    history=[]
    for _ in range(steps):
        logits=scores(model,batch,encode,pad_id)
        if mode=='sft':loss=nn.functional.cross_entropy(logits,target)
        else:
            dist=torch.distributions.Categorical(logits=logits);a=dist.sample((8,))
            observed=rewards[torch.arange(len(batch),device=device)[None,:],a]
            baseline=(dist.probs.detach()*rewards).sum(-1)
            pg=-((observed-baseline)*dist.log_prob(a)).mean()
            logp=logits.log_softmax(-1);kl=(logp.exp()*(logp-reference)).sum(-1).mean()
            loss=pg+.01*kl
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        o.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);o.step();history.append(float(loss.detach()))
    return model.eval(),history


def evaluate(model,batch,encode=ascii_encode,pad_id=0):
    with torch.no_grad():idx=scores(model,batch,encode,pad_id).argmax(-1).cpu().tolist()
    texts=[CANDIDATES[i] for i in idx]
    return {'semantic_reward':float(np.mean([reward(t,c) for t,c in zip(batch,texts)])),
            'exact_template_accuracy':float(np.mean([t['target']==c for t,c in zip(batch,texts)])),
            'solver_completion_rate':float(np.mean([solve(t,c) is not None for t,c in zip(batch,texts)]))}


def load_local_hf(path):
    # Optional real causal-LM backend. No downloads, remote code, API keys, or unrequested services.
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:raise ValueError('Tokenizer needs a padding or EOS token')
    return model,lambda s:tok.encode(s,add_special_tokens=False),pad


def benchmark():
    torch.set_num_threads(1);rows=[]
    for seed in (51,52,53):
        torch.manual_seed(seed);base=TinyLM();batch=tasks(11,16)
        for mode in ('untrained','sft','rl'):
            m=copy.deepcopy(base)
            if mode!='untrained':m,_=train(m,batch,mode,steps=50,seed=seed)
            rows.append({'seed':seed,'method':mode,'nominal':evaluate(m,tasks(99,16)),
                         'numeric_shift':evaluate(m,tasks(999,16,shift=True))})
    return {'scope':'tiny character GRU, 8 fixed DSL completions; NOT a trained large language model',
            'hf_backend':'implemented but not executed locally; requires local checkpoint and transformers',
            'reward_semantics':'known synthetic oracle plus feasibility probes; not general semantic verification',
            'results':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results.json');p.add_argument('--local-model')
    args=p.parse_args()
    if args.local_model:
        m,encode,pad=load_local_hf(args.local_model);m,h=train(m,tasks(11,16),encode=encode,pad_id=pad)
        m.save_pretrained(str(Path(args.output).with_suffix(''))+'-checkpoint',safe_serialization=True)
        result={'backend':'local_hf','evaluation':evaluate(m,tasks(99,16),encode,pad)}
    else:result=benchmark()
    Path(args.output).write_text(json.dumps(result,indent=2))
