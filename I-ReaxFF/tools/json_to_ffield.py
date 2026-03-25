#!/usr/bin/env python
import json as js

from irff.reaxfflib import write_ffield


def jsontoffield():
    with open('ffield.json', 'r') as lf:
        j = js.load(lf)
    p_ = j['p']
    m_ = j['m']
    mf_layer = j.get('mf_layer')
    be_layer = j.get('be_layer')

    spec, bonds, offd, angs, torp, hbs = init_bonds(p_)

    # Empty dict means no neural-network payload in many li-style JSONs.
    if isinstance(m_, dict) and not m_:
        m_ = None
        mf_layer = None
        be_layer = None

    if m_ and spec and not all(f'fmwo_{sp}' in m_ for sp in spec):
        print(
            "-  Warning: neural network weights missing expected keys, dropping m/mf_layer/be_layer",
        )
        m_ = None
        mf_layer = None
        be_layer = None
    libfile = 'ffield'
    meta_path = f'{libfile}.meta.json'
    with open(meta_path, 'w', encoding='utf-8') as mf:
        js.dump(j, mf, sort_keys=True, indent=2)

    while True:
        try:
            write_ffield(
                p_,
                spec,
                bonds,
                offd,
                angs,
                torp,
                hbs,
                m=m_,
                mf_layer=mf_layer,
                be_layer=be_layer,
                libfile=libfile,
            )
            break
        except KeyError as exc:
            missing = exc.args[0]
            if missing in p_:
                raise
            print(f"-  Warning: missing parameter {missing}, defaulting to 0.0")
            p_[missing] = 0.0


def init_bonds(p_):
    spec,bonds,offd,angs,torp,hbs = [],[],[],[],[],[]
    for key in p_:
        # key = key.encode('raw_unicode_escape')
        # print(key)
        k = key.split('_')
        if k[0]=='bo1':
           bonds.append(k[1])
        elif k[0]=='rosi':
           kk = k[1].split('-')
           # print(kk)
           if len(kk)==2:
              if kk[0]!=kk[1]:
                 offd.append(k[1])
           elif len(kk)==1:
              spec.append(k[1])
        elif k[0]=='theta0':
           angs.append(k[1])
        elif k[0]=='tor1':
           torp.append(k[1])
        elif k[0]=='rohb':
           hbs.append(k[1])
    return spec,bonds,offd,angs,torp,hbs


if __name__ == '__main__':
    jsontoffield()
