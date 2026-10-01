"""Finite indexed linear IR with units, table lineage, lag expansion, and solver audits."""
from __future__ import annotations
import argparse
import copy
import csv
import itertools
import json
import re
from pathlib import Path
import numpy as np
from scipy.optimize import milp,linprog,Bounds,LinearConstraint


def unit(u):
    if not isinstance(u,dict) or any(not isinstance(k,str) or type(v) is not int for k,v in u.items()):
        raise ValueError('Units are maps of base dimensions to integer exponents')
    return {k:v for k,v in u.items() if v}


def multiply(u,v):
    w=dict(unit(u))
    for k,n in unit(v).items():w[k]=w.get(k,0)+n
    return unit(w)


def table(path,axes,dimension):
    """Load a numeric CSV parameter; preserve exact file/row/value-column lineage."""
    values={};sources={}
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f)
        if not set(axes+['value']).issubset(reader.fieldnames or []):raise ValueError('CSV headers')
        for i,row in enumerate(reader,2):
            key='|'.join(row[a] for a in axes);v=float(row['value'])
            if key in values or not np.isfinite(v):raise ValueError('Duplicate/nonfinite parameter')
            values[key]=v;sources[key]={'file':str(path),'row':i,'column':'value'}
    return {'axes':axes,'unit':unit(dimension),'values':values,'sources':sources}


def contexts(bindings,sets,base=None):
    base={} if base is None else dict(base)
    if set(bindings)&set(base):raise ValueError('Shadowed index binding')
    for vals in itertools.product(*(sets[a] for a in bindings.values())):
        yield base|dict(zip(bindings,vals))


def compile_spec(spec):
    allowed={'sets','parameters','variables','constraints','objective'}
    if set(spec)!=allowed:raise ValueError('Unknown or missing top-level fields')
    sets=spec['sets'];params=spec['parameters'];variables=spec['variables']
    for name,labels in sets.items():
        if (not re.fullmatch(r'[A-Za-z_]\w*',name) or not labels or len(set(labels))!=len(labels)
            or any(not isinstance(v,str) or '|' in v for v in labels)):
            raise ValueError('Sets require unique string labels without |')
    for _name,p in params.items():
        expected={'|'.join(i) for i in itertools.product(*(sets[a] for a in p['axes']))}
        if set(p['values'])!=expected or not all(np.isfinite(v) for v in p['values'].values()):
            raise ValueError('Parameter table does not match declared index domain')
        unit(p['unit'])
    keys=[];dims={};lb=[];ub=[];integrality=[]
    for name,v in variables.items():
        if v['kind'] not in ('continuous','integer','binary'):raise ValueError('Variable kind')
        dims[name]=unit(v['unit'])
        low=float(v.get('lb',0));high=np.inf if v.get('ub') is None else float(v['ub'])
        if not np.isfinite(low) or np.isnan(high) or low>high:raise ValueError('Variable bounds')
        if v['kind']=='binary':low=max(low,0);high=min(high,1)
        if low>high:raise ValueError('Empty binary domain')
        for idx in itertools.product(*(sets[a] for a in v['axes'])):
            keys.append((name,)+idx);lb.append(low);ub.append(high);integrality.append(v['kind']!='continuous')
    if not keys or len(keys)>5000:raise ValueError('Expanded variable count must be 1..5000')
    pos={k:i for i,k in enumerate(keys)};lineage=[]
    def indices(tokens,ctx,binding_axes):
        ans=[]
        for tok in tokens:
            if isinstance(tok,dict):
                if set(tok)!={'previous'}:raise ValueError('Unsupported index expression')
                b=tok['previous'];domain=sets[binding_axes[b]];i=domain.index(ctx[b])
                if i==0:return None
                ans.append(domain[i-1])
            elif isinstance(tok,str) and tok.startswith('@'):ans.append(ctx[tok[1:]])
            else:ans.append(tok)
        return tuple(ans)
    def scalar(expr,ctx,axes):
        if isinstance(expr,(int,float)) and not isinstance(expr,bool):
            if not np.isfinite(expr):raise ValueError('Nonfinite scalar')
            return float(expr),{}
        if not isinstance(expr,dict) or set(expr)!={'parameter','index'}:raise ValueError('Scalar syntax')
        p=params[expr['parameter']];idx=indices(expr['index'],ctx,axes)
        if idx is None:raise ValueError('Parameter lag has no boundary value')
        key='|'.join(idx)
        if key not in p['values']:raise ValueError('Unknown parameter cell')
        lineage.append({'parameter':expr['parameter'],'key':key,'source':p.get('sources',{}).get(key)})
        return float(p['values'][key]),unit(p['unit'])
    def expression(terms,ctx,axes,target_unit):
        row=np.zeros(len(keys))
        for term in terms:
            if not set(term).issubset({'var','index','coef','sum','omit_at_boundary'}):raise ValueError('Term syntax')
            sums=term.get('sum',{});all_axes=axes|sums
            for local in contexts(sums,sets,ctx):
                idx=indices(term['index'],local,all_axes)
                if idx is None:
                    if term.get('omit_at_boundary') is True:continue
                    raise ValueError('Undeclared lag boundary')
                key=(term['var'],)+idx
                if key not in pos:raise ValueError('Unknown indexed variable')
                value,dim=scalar(term.get('coef',1),local,all_axes)
                if multiply(dim,dims[term['var']])!=unit(target_unit):raise ValueError('Unit mismatch in term')
                row[pos[key]]+=value
        return row
    rows=[];lows=[];highs=[];names=[]
    for con in spec['constraints']:
        axes=con.get('foreach',{})
        for ctx in contexts(axes,sets):
            name=con['name']+str(tuple(ctx.items()))
            if name in names:raise ValueError('Duplicate expanded constraint')
            names.append(name);rows.append(expression(con['terms'],ctx,axes,con['unit']))
            rhs,dim=scalar(con['rhs'],ctx,axes)
            if dim!=unit(con['unit']) and rhs!=0:raise ValueError('RHS unit mismatch')
            sense=con['sense']
            if sense not in ('<=','>=','='):raise ValueError('Constraint sense')
            lows.append(rhs if sense in ('>=','=') else -np.inf)
            highs.append(rhs if sense in ('<=','=') else np.inf)
    obj=spec['objective']
    if obj['sense'] not in ('min','max'):raise ValueError('Objective sense')
    c=expression(obj['terms'],{},{},obj['unit']);sign=1 if obj['sense']=='min' else -1
    return {'c':c*sign,'A':np.array(rows).reshape(-1,len(keys)),'lower':np.array(lows),
            'upper':np.array(highs),'lb':np.array(lb),'ub':np.array(ub),'integer':np.array(integrality),
            'keys':keys,'lineage':lineage,'sign':sign}


def solve(spec):
    m=compile_spec(spec)
    r=milp(m['c'],integrality=m['integer'],bounds=Bounds(m['lb'],m['ub']),
           constraints=LinearConstraint(m['A'],m['lower'],m['upper']),
           options={'time_limit':10.,'mip_rel_gap':0.0})
    if r.status!=0 or r.x is None:raise RuntimeError(f'Unproven solve: {r.message}')
    v=m['A']@r.x
    err=max(0.,np.max(m['lb']-r.x),np.max(r.x-m['ub']),
            np.max(m['lower']-v,initial=0),np.max(v-m['upper'],initial=0),
            np.max(np.abs(r.x[m['integer']]-np.round(r.x[m['integer']])),initial=0))
    if err>1e-6:raise RuntimeError('Independent solution audit failed')
    return {'objective':float(m['sign']*r.fun),'max_violation':float(err),
            'variables':len(r.x),'constraints':len(v),'lineage':m['lineage']}


def production_spec(periods=3):
    T=[str(i) for i in range(periods)];item={'item':1};money={'currency':1}
    P=lambda name:{'parameter':name,'index':['@t']}
    make={'var':'make','index':['@t'],'coef':1}
    return {'sets':{'T':T},'parameters':{
        'demand':{'axes':['T'],'unit':item,'values':{t:float(2+int(t)%2) for t in T}},
        'capacity':{'axes':['T'],'unit':item,'values':{t:5. for t in T}},
        'price':{'axes':['T'],'unit':{'currency':1,'item':-1},'values':{t:1.+int(t) for t in T}},
        'holding':{'axes':[],'unit':{'currency':1,'item':-1},'values':{'':.1}}},
        'variables':{v:{'axes':['T'],'unit':item,'kind':'continuous','lb':0,'ub':None} for v in ('make','stock')},
        'constraints':[
            {'name':'balance','foreach':{'t':'T'},'unit':item,'sense':'=','rhs':P('demand'),
             'terms':[make,{'var':'stock','index':['@t'],'coef':-1},
                      {'var':'stock','index':[{'previous':'t'}],'coef':1,'omit_at_boundary':True}]},
            {'name':'capacity','foreach':{'t':'T'},'unit':item,'sense':'<=','rhs':P('capacity'),'terms':[make]}],
        'objective':{'sense':'min','unit':money,'terms':[
            make|{'sum':{'t':'T'},'coef':P('price')},
            {'var':'stock','index':['@t'],'sum':{'t':'T'},'coef':{'parameter':'holding','index':[]}}]}}


def reference(periods):
    # Hand-built independently of the indexed compiler.
    c=np.r_[1+np.arange(periods),np.full(periods,.1)];A=np.zeros((periods,2*periods))
    for t in range(periods):
        A[t,t]=1;A[t,periods+t]=-1
        if t:A[t,periods+t-1]=1
    r=linprog(c,A_eq=A,b_eq=2+np.arange(periods)%2,
              bounds=[(0,5)]*periods+[(0,None)]*periods,method='highs')
    if not r.success:raise RuntimeError(r.message)
    return float(r.fun)


def benchmark():
    rows=[]
    for T in (2,3,5,8):
        s=production_spec(T);r=solve(s);baseline=reference(T)
        if abs(r['objective']-baseline)>1e-6:raise RuntimeError('Reference disagreement')
        bad=copy.deepcopy(s);bad['constraints'][0]['terms'].pop()
        rows.append({'periods':T,'correct_objective':r['objective'],'reference':baseline,
                     'missing_carryover_objective':solve(bad)['objective']})
    return {'scope':'typed indexed LP/MILP compiler, NOT general natural-language semantic proof','results':rows}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results.json')
    args=p.parse_args();Path(args.output).write_text(json.dumps(benchmark(),indent=2))
