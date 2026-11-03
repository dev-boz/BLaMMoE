import sys, json, time, hashlib, copy, math
from pathlib import Path
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
sys.path.insert(0, '/home/boz/BLaMMoE/scripts')
from blammoe.toy import setup
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.fused import FusedPair, run_pair, char_predictions, first_char_matrix, _js, bits_per_doc, BusAblation, NO_ABLATION
from blammoe.toy.fused_train import domain_eval_sets
from blammoe.toy.batching import make_batch
from blammoe.toy.tokenizers import CharTokenizer
from blammoe.toy.moe import load
from blammoe.stats import paired_bootstrap_diff
from toy_partner_shift import doc_rows, pooled
WORK=Path('/tmp/claude-1000/-home-boz/0862934b-3b4c-4087-9ca1-94a01b1b355d/scratchpad/codex/work')

def getpair(root,unit,signal):
    path=Path('results')/root/unit
    cfg=ToyPOCConfig.from_yaml(path/'config.yaml')
    for spec in (cfg.model_a,cfg.model_b):
        if spec.tokenizer=='bpe': assert (path/setup.bpe_name(spec,cfg)).exists()
    toks=[setup.tokenizer(s,cfg) for s in (cfg.model_a,cfg.model_b)]
    torch.manual_seed(cfg.fused.seed)
    A,_=load(path/'model_a.pt'); B,_=load(path/'model_b.pt')
    v='ab_'+signal+'_all_rel_safe_text'
    pair=FusedPair(A,B,[0,1,2,3],cfg.fused.d_bus,cfg.fused.n_heads,signal,True,True).eval()
    init_hash=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in pair.links.state_dict().values())).hexdigest()
    pair.links.load_state_dict(torch.load(path/f'fused_{v}.pt',map_location='cpu',weights_only=True))
    return pair,cfg,toks,init_hash

@torch.no_grad()
def metrics(root,unit,signal):
    pair,cfg,toks,ih=getpair(root,unit,signal)
    chars=CharTokenizer().itos[1:]
    mats=[first_char_matrix(t,chars) for t in toks]
    out={'init_hash':ih,'domains':{}}
    for domain,docs in domain_eval_sets(*setup.texts(cfg,'test'),cfg.seq_chars).items():
        batches=[make_batch(docs,t,cfg.seq_chars) for t in toks]
        conds={'solo':{'disabled':frozenset(range(4))},'fused':{},'wrong':{'partner_roll':1}}
        preds={}; scores={}; raw={}
        for c,kw in conds.items():
            logits=run_pair(pair,*batches,**kw)
            raw[c]=logits
            preds[c]=[char_predictions(l,b,m) for l,b,m in zip(logits,batches,mats)]
            scores[c]=[pooled(bits_per_doc(l,b)) for l,b in zip(logits,batches)]
        data=[]
        for side in (0,1):
            keys=list(set(preds['solo'][side]) & set(preds['solo'][1-side]))
            P= torch.stack([preds['solo'][1-side][k] for k in keys])
            S,F,W=[torch.stack([preds[c][side][k] for k in keys]) for c in ('solo','fused','wrong')]
            js0,jsf,jsw=(_js(q,P) for q in (S,F,W))
            rows=[doc_rows(x,[k[0] for k in keys],len(docs)) for x in (js0,jsf,jsw)]
            data.append({'solo_to_partner':float(js0.mean()),'fused_to_partner':float(jsf.mean()),
                'toward':float((jsw-jsf).mean()),'actual_attraction':paired_bootstrap_diff(rows[0],rows[1]),
                'wrong_repulsion':float((jsw-js0).mean()),'pad_mass_max':float(1-S.sum(-1).min()),
                'bits':{c:scores[c][side] for c in conds}})
        out['domains'][domain]=data
        print(root,unit,signal,domain,json.dumps(data),flush=True)
        if root=='toy_char_pair' and unit=='e0':
            extras={}
            for name,kw in [('only_last',{'disabled':frozenset([0,1,2])}),('without_last',{'disabled':frozenset([3])})]:
                logits=run_pair(pair,*batches,**kw)
                extras[name]=[pooled(bits_per_doc(l,b)) for l,b in zip(logits,batches)]
            # A and B are paired consistently by permuting B ONCE at input.
            idx=[(i-1)%len(docs) for i in range(len(docs))]
            pb=make_batch([docs[i] for i in idx],toks[1],cfg.seq_chars)
            la,lb=run_pair(pair,batches[0],pb)
            extras['consistent_wrong']=[pooled(bits_per_doc(la,batches[0])),pooled(bits_per_doc(lb,pb))]
            if signal=='experts':
                for name,abl in [('tags_zero',BusAblation(tags='zero')),
                                 ('content_zero',BusAblation(blank=frozenset(range(8))))]:
                    la,lb=run_pair(pair,*batches,ablate=(abl,abl))
                    extras[name]=[pooled(bits_per_doc(l,b)) for l,b in zip((la,lb),batches)]
            else:
                # Log actual injected activation sizes, not only scalar gates.
                norms={}; handles=[]
                for depth,link in pair.links.items():
                    for side,rx in [('a',link.rx_a),('b',link.rx_b)]:
                        def hook(mod,args,output,key=(depth,side)):
                            x=args[0]; delta=output-x
                            norms[str(key)]={'gate':float(mod.gate.tanh()),'rms_delta':float(delta.square().mean().sqrt()),'rms_x':float(x.square().mean().sqrt())}
                        handles.append(rx.register_forward_hook(hook))
                run_pair(pair,*batches)
                for h in handles:h.remove()
                extras['injection']=norms
            out['domains'][domain+'_extras']=extras
            print('EXTRAS',signal,domain,json.dumps(extras),flush=True)
    return out

if __name__=='__main__':
    out={}; start=time.time()
    for root,units,signals in [('toy_char_pair',['e0','e1','e2'],['experts','resid']),('toy_complement',['s0','s1','s2'],['experts'])]:
        for unit in units:
            for signal in signals:
                key=f'{root}/{unit}/{signal}'; out[key]=metrics(root,unit,signal)
                (WORK/'metrics.json').write_text(json.dumps(out,indent=2))
    print('elapsed',time.time()-start,flush=True)
