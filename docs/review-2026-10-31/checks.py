import json, random, hashlib, sys, time, bisect
from pathlib import Path
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
sys.path.insert(0,'/home/boz/BLaMMoE/scripts')
from blammoe.toy import setup
from blammoe.toy.audit import causality_gate
from blammoe.toy.corpus import Sample, ascii_clean, split_names, MixedSampler
from blammoe.toy.batching import make_batch
from blammoe.toy.fused import FusedPair, _js, run_pair, bits_per_doc, pair_loss
from blammoe.toy.moe import load, ToyMoELM, ToyMoEConfig
from blammoe.toy.tokenizers import CharTokenizer
from blammoe.toy.fused_train import domain_eval_sets
from blammoe.toy.config import ToyPOCConfig
from blammoe.toy.sweep import units, SweepConfig
import toy_partner_shift as ps
import toy_where_knowledge as wk
import yaml
WORK=Path('/tmp/claude-1000/-home-boz/0862934b-3b4c-4087-9ca1-94a01b1b355d/scratchpad/codex/work')

def pair_for(unit,signal):
    p=Path('results/toy_char_pair')/unit
    cfg=ToyPOCConfig.from_yaml(p/'config.yaml')
    A,_=load(p/'model_a.pt'); B,_=load(p/'model_b.pt')
    pair=FusedPair(A,B,[0,1,2,3],64,4,signal,True,True).eval()
    pair.links.load_state_dict(torch.load(p/f'fused_ab_{signal}_all_rel_safe_text.pt',map_location='cpu',weights_only=True))
    return pair,cfg

out={}; start=time.time()
# Pair configurations and file contents, read-only.
sc=SweepConfig.from_yaml('configs/toy_char_pair.yaml')
out['copies']=[]
for u in units(sc,yaml.safe_load(Path(sc.base).read_text())):
    if not u.name.startswith(('e','f')):continue
    for dest,src in u.copy_models.items():
        got=hashlib.sha256((u.out/dest).read_bytes()).hexdigest()
        want=hashlib.sha256(src.read_bytes()).hexdigest()
        out['copies'].append([u.name,dest,str(src),got==want])
print('COPIES',out['copies'],flush=True)
# A falsifying example for the proposed metric: real partner changes nothing,
# wrong partner damages the output. This still meets dependence+toward > 0.
S=torch.tensor([0.9,0.1]); P=S.clone(); W=torch.tensor([0.1,0.9])
r=ps.shift_rows({'solo':{(0,1):S},'fused':{(0,1):S},'wrong':{(0,1):W}}, {(0,1):P},['aa'],['a','b'])
out['toward_null']={k:ps.pooled(v) for k,v in r.items()}
print('TOWARD_NULL',out['toward_null'],flush=True)
# The actual number of independent source papers behind the 48 prose windows.
files=sorted(Path('papers').glob('*.md'))
test_names=split_names([p.name for p in files],0.1,0.1)['test']
parts=[(p.name,ascii_clean(p.read_text(errors='ignore'))) for p in files if p.name in test_names]
length=sum(len(t) for _,t in parts)+len(parts)-1
rr=random.Random(778); counts={n:0 for n,t in parts}; bounds=[]; e=0
for n,t in parts:e+=len(t)+1; bounds.append(e)
starts=[]
for _ in range(48):
    i=rr.randrange(0,length-256); starts.append(i); counts[parts[bisect.bisect_right(bounds,i)][0]]+=1
out['prose_sources']={'length':length,'counts':counts,'overlapping_pairs':sum(abs(x-y)<256 for j,x in enumerate(starts) for y in starts[:j])}
print('PROSE_SOURCES',out['prose_sources'],flush=True)

pair,cfg=pair_for('e0','resid'); pair.double()
tok=CharTokenizer()
docs=domain_eval_sets(*setup.texts(cfg,'test'),cfg.seq_chars,n=1,seed=1778)
checkdocs=[(x.text,x.text) for dd in docs.values() for x in dd]
out['trained_resid_audit']=causality_gate(pair,checkdocs,tok,tok,cfg.seq_chars,tol=1e-9,batch=16)
print('TRAINED_RESID_AUDIT',out['trained_resid_audit'],flush=True)
# Direct suffix intervention with the trained residual model, both streams varied
# separately, checking all vocabulary probabilities at every unchanged char.
with torch.no_grad():
    text=checkdocs[0][0]; c=97
    def scores(a,b):
        bs=[make_batch([Sample(text=x,blind=x)],tok,256) for x in (a,b)]
        return [y[0,:c].log_softmax(-1) for y in run_pair(pair,*bs)]
    base=scores(text,text); alt=text[:c]+''.join(random.Random(42+i).choice('az 019,;\n') for i in range(len(text)-c))
    out['suffix_tests']={}
    for name,ts in [('a_only',(alt,text)),('b_only',(text,alt)),('both',(alt,alt))]:
        vals=scores(*ts); out['suffix_tests'][name]=[float((a-b).abs().max()) for a,b in zip(base,vals)]
print('SUFFIX_TESTS',out['suffix_tests'],flush=True)
# Fail-open behaviour on nonfinite predictions (tiny pair, no checkpoint edits).
tinycfg=ToyMoEConfig(tok.vocab_size,d_model=8,n_layers=1,n_heads=2,n_experts=2,d_expert=8,max_len=32)
tiny=FusedPair(ToyMoELM(tinycfg),ToyMoELM(tinycfg),[],8,2,'resid',False,True).double().eval()
forward=tiny.forward
tiny.forward=lambda *a,**k:tuple(x*float('nan') for x in forward(*a,**k))
out['nan_audit']=causality_gate(tiny,[('abcd','abcd')],tok,tok,32)
print('NAN_AUDIT',out['nan_audit'],flush=True)

# Re-evaluate the char-pair on a fresh set of windows.
out['fresh_eval']={}
for signal in ('experts','resid'):
    pair,cfg=pair_for('e0',signal)
    out['fresh_eval'][signal]={}
    for dom,docs in domain_eval_sets(*setup.texts(cfg,'test'),256,48,1778).items():
        b=make_batch(docs,tok,256)
        with torch.no_grad():
            solo=run_pair(pair,b,b,disabled=frozenset(range(4)))
            fused=run_pair(pair,b,b)
            out['fresh_eval'][signal][dom]={'solo':[ps.pooled(bits_per_doc(l,b)) for l in solo], 'fused':[ps.pooled(bits_per_doc(l,b)) for l in fused]}
        print('FRESH',signal,dom,out['fresh_eval'][signal][dom],flush=True)

# Verify the single-expert ablation against directly replacing that expert's
# routed output, using a trained BPE model (and its real routing weights).
path=Path('results/toy_complement/s0'); c=ToyPOCConfig.from_yaml(path/'config.yaml')
assert (path/setup.bpe_name(c.model_b,c)).exists()
tb=setup.tokenizer(c.model_b,c); B,_=load(path/'model_b.pt')
ss=MixedSampler(*setup.texts(c,'train'),256,p_recall=0,seed=4321)
ref=make_batch([ss.sample(d) for d in ('prose','code') for _ in range(8)],tb,256)
b=make_batch(domain_eval_sets(*setup.texts(c,'test'),256,n=4)['prose'],tb,256)
with torch.no_grad():
    means=wk.reference_means(B,ref)
    with wk.Replace(wk.ablation(B,means,('expert',1,2))): got=B(b.ids)
    ex=B.layers[1].moe.experts[2]; orig=ex.forward
    ex.forward=lambda x:means['expert',1,2].expand(x.shape[0],-1)
    want=B(b.ids); ex.forward=orig
    out['trained_expert_ablation_max_logit_error']=float((got-want).abs().max())
print('TRAINED_ABLATION',out['trained_expert_ablation_max_logit_error'],flush=True)

out['elapsed_seconds']=time.time()-start
(WORK/'checks.json').write_text(json.dumps(out,indent=2))
print('ELAPSED',out['elapsed_seconds'],flush=True)
