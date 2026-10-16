#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""One independent image target per request and precision, never per topology.

Ordinary requests inherit the published golden spec. Same-trial reference
images are reused by bytes; extra seed/size requests use the identical isolated
diffusers recipe, once, in a locked pool shared by TP1/TP2/SP2/batching. Neither
native images nor candidate execution layout enter the reference key.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILE = 'mixed'
SCHEMA = 'sd3-fixed-request-reference-v1'
_METRICS = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixtures():
    return json.loads((HERE / 'sd3_golden.json').read_text())['fixtures']


def request(fid, *, seed=None, cfg=None, height=None, width=None, **extra):
    f = next(f for f in fixtures() if f['id'] == fid)
    if set(extra) - {'num_outputs_per_prompt'}:
        raise ValueError('Only output multiplicity may be added to a canonical request')
    return dict(prompt=f['prompt'], seed=int(f['seed'] if seed is None else seed),
                guidance_scale=float(f['cfg'] if cfg is None else cfg),
                num_inference_steps=int(f['steps']), height=int(f['h'] if height is None else height),
                width=int(f['w'] if width is None else width), **extra)


def fixture_for(req, index=0):


    f = next((f for f in fixtures() if f['prompt'] == req['prompt']
              and float(f['cfg']) == float(req['guidance_scale'])), {})
    return dict(id=f.get('id', 'extended_request'), prompt=req['prompt'],
                seed=int(req['seed']) + index, cfg=float(req['guidance_scale']),
                steps=int(req['num_inference_steps']), h=int(req['height']), w=int(req['width']))


def semantics(fx):
    return {k: fx[k] for k in ('prompt','seed','cfg','steps','h','w')}


def load_reference(work, reqs, model, artifacts, profile=PROFILE):
    import fcntl
    import numpy as np
    import generate_goldens_inrun as GI

    if profile not in GI.REF_SETS:
        raise ValueError('Unknown reference precision')
    run_id = os.environ.get('SD3_VERIFIER_RUN_ID')
    if not run_id:
        raise ValueError('Fixed reference reuse requires the current verifier run id')
    recipe = {'schema': SCHEMA, 'run_id': run_id, 'profile': profile, 'model': model,
              'generator_sha256': hashlib.sha256(GI._gen_script().encode()).hexdigest(),
              'precision_sha256': file_sha(HERE / 'sd3_precision_profiles.py'),
              'fixture_spec_sha256': file_sha(HERE / 'sd3_golden.json'), 'negative_prompt': ' '}
    root = Path(artifacts) / 'sd3_fixed_reference' / digest(recipe)
    root.mkdir(parents=True, exist_ok=True)
    desired = {}
    for key, req in reqs.items():
        desired[key] = [fixture_for(req, i) for i in range(int(req.get('num_outputs_per_prompt', 1)))]

    unique = {digest(semantics(f)): f for fs in desired.values() for f in fs}
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / 'manifest.json'
        manifest = json.loads(path.read_text()) if path.exists() else {'recipe': recipe, 'images': {}}
        if manifest['recipe'] != recipe:
            raise ValueError('Reference recipe mismatch')
        gold = os.environ.get('SD3_GOLDEN_DIR')
        source = Path(gold) / GI.REF_SETS[profile][1] if gold else None
        published = {}
        if source and (source / 'sd3_golden.json').is_file():
            meta = json.loads((source / 'sd3_golden.json').read_text())


            if meta.get('verifier_run_id') == run_id:
                if meta.get('precision', {}).get('set_dtype') != GI.REF_SETS[profile][0]:
                    raise ValueError('In-run reference precision mismatch')
                published = {digest(semantics(f)): f for f in meta['fixtures']}
        missing = []
        for key, fx in unique.items():
            dst = root / f'{key}.npy'
            if key in manifest['images']:
                if not dst.is_file() or file_sha(dst) != manifest['images'][key]['sha256']:
                    raise ValueError('Frozen reference bytes changed')
                continue
            if key in published:
                f = published[key]
                src = source / f['golden_npy']
                if src.parent.resolve() != source.resolve():
                    raise ValueError('Reference path escaped its directory')
                shutil.copyfile(src, dst)
                manifest['images'][key] = {'sha256': file_sha(dst), 'request': semantics(fx),
                                          'origin': str(src), 'origin_sha256': file_sha(src)}
            else:
                missing.append({**fx, 'id': key, 'golden_npy': f'{key}.npy', 'shape': [fx['h'],fx['w'],3]})
        if missing:


            child = root / ('generate_' + digest([semantics(f) for f in missing]))
            child.mkdir(exist_ok=True)
            (child / 'spec.json').write_text(json.dumps({'fixtures': missing, 'smoke_ids': [], 'diagnostic_ids': []}))
            ws = Path(os.environ.get('INFB_WORKSPACE', '/app')).resolve()
            env = {k:v for k,v in os.environ.items() if not k.startswith('INFB_SD3_')}
            env['PYTHONPATH'] = os.pathsep.join(q for q in env.get('PYTHONPATH','').split(os.pathsep)
                if q and not Path(q).resolve().is_relative_to(ws) and not Path(q).resolve().is_relative_to(HERE))
            env.update(SD3_SPEC=str(child/'spec.json'), SD3_MODEL=model, SD3_OUT=str(child),
                       SD3_REF_DTYPE=GI.REF_SETS[profile][0], SD3_NEG=' ',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            with (child/'log.txt').open('w') as log:
                proc = subprocess.run([os.environ.get('SD3_PYTHON',sys.executable), '-P','-B','-c',GI._gen_script()],
                    cwd=child, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            if proc.returncode:
                raise RuntimeError(f'Independent reference failed: rc={proc.returncode}; {child}/log.txt')
            for fx in missing:
                key=fx['id']; src=child/f'{key}.npy'; dst=root/f'{key}.npy'
                shutil.copyfile(src,dst)
                manifest['images'][key]={'sha256':file_sha(dst),'request':semantics(fx),
                                         'origin':str(src),'origin_sha256':file_sha(src)}
        for key, fx in unique.items():
            arr=np.load(root/f'{key}.npy', allow_pickle=False)
            if arr.dtype != np.uint8 or arr.shape != (fx['h'],fx['w'],3):
                raise ValueError('Independent reference dtype/shape mismatch')
        tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(manifest,indent=2)); os.replace(tmp,path)
    receipt={'schema':SCHEMA,'profile':profile,'recipe':recipe,'pool':str(root),'requests':{},'rc':0}
    images={}
    for name, fs in desired.items():
        keys=[digest(semantics(f)) for f in fs]
        images[name]=[np.load(root/f'{key}.npy',allow_pickle=False) for key in keys]
        receipt['requests'][name]=[{'key':key,'path':str(root/f'{key}.npy'),**manifest['images'][key]} for key in keys]
    Path(work).mkdir(parents=True,exist_ok=True)
    (Path(work)/'fixed_reference.json').write_text(json.dumps(receipt,indent=2))
    return {**receipt,'images':images}


def measure(image, reference, prompt):


    import verify_callsite_main as V
    key=(image.shape,reference.shape,hashlib.sha256(image.tobytes()).hexdigest(),
         hashlib.sha256(reference.tobytes()).hexdigest(),prompt)
    if key in _METRICS:
        return dict(_METRICS[key])
    ie,te=V._clip_embeds([image,reference],[prompt])
    metrics={'psnr':V.psnr(image,reference),'ssim':V.ssim(image,reference),
             'clip_img':float((ie[0]*ie[1]).sum()),'clip_txt':float((ie[0]*te[0]).sum())}
    if len(_METRICS)<256:
        _METRICS[key]=dict(metrics)
    return metrics


def image_check(image, reference, req, index=0, profile=PROFILE):
    import numpy as np
    import verify_callsite_main as V
    if reference is None or req is None:
        return {'verdict':'not_evaluated','reason':'independent request reference missing'}
    fx=fixture_for(req,index)
    bars=V.profile_bars_for(fx,profile)
    out={'profile':profile,'fixture':fx,'bars':bars,'metrics':{},
         'threshold_policy':V._IMAGE_BARS.receipt()}
    shape=(fx['h'],fx['w'],3)
    if np.asarray(reference).dtype != np.uint8 or np.asarray(reference).shape != shape:
        return {**out,'verdict':'not_evaluated','reason':'reference dtype/shape invalid'}
    if image is None or np.asarray(image).dtype != np.uint8 or np.asarray(image).shape != shape:
        return {**out,'verdict':'fail','reason':'candidate image missing or dtype/shape invalid'}
    try:
        values=measure(image,reference,fx['prompt'])
        out['metrics']=values
    except Exception as exc:
        return {**out,'verdict':'not_evaluated','reason':f'image scorer failed: {type(exc).__name__}: {exc}'[:500]}
    if any(type(values.get(k)) not in (float,int) or not math.isfinite(values[k]) for k in bars):
        return {**out,'verdict':'not_evaluated','reason':'image scorer has missing/nonfinite metrics'}
    bad=[k for k in bars if values[k]<bars[k]]
    return {**out,'verdict':'fail' if bad else 'pass','reason':','.join(bad)}


def check_outputs(reading,key,images):
    req=(reading.get('requests') or {}).get(key)
    refs=(reading.get('reference') or {}).get(key) or []
    if req is None:
        return {'verdict':'not_evaluated','reason':f'{key}: request spec missing','per':[]}
    n=int(req.get('num_outputs_per_prompt',1))
    if len(refs)!=n or any(ref is None for ref in refs):
        return {'verdict':'not_evaluated','reason':f'{key}: incomplete independent reference','per':[]}
    if len(images or [])!=n:
        return {'verdict':'fail','reason':f'{key}: expected {n} returned images','per':[]}
    per=[image_check(img,refs[i],req,i) for i,img in enumerate(images)]
    verdict='fail' if any(x['verdict']=='fail' for x in per) else (
        'not_evaluated' if any(x['verdict']=='not_evaluated' for x in per) else 'pass')
    return {'verdict':verdict,'reason':f'{key}: '+ '; '.join(x.get('reason','') for x in per if x['verdict']!='pass'),'per':per}
