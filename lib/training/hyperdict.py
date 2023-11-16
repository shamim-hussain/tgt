from types import SimpleNamespace

class HDict(dict):
    class L:
        def __init__(self, func):
            self.func = func
        def __call__(self, base):
            return self.func(base)
    
    class R:
        def __init__(self, meta=None):
            self.meta = meta

    @property
    def P(self):
        try:
            return self.__dict__['__P__']
        except KeyError:
            return AttributeError('Dictionary has no parent')

    def __dir__(self):
        return super().__dir__() + list(self.keys())
    
    def __setattr__(self, key, value):
        if key in self.keys():
            self[key]=value
        else:
            raise AttributeError('No such attribute: '+key)
    
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            node = self.__getnode__(key)
            if node:
                return node
            raise AttributeError('No such attribute: '+key)
        
        if isinstance(value, self.__class__.L):
            return value(self)
        elif isinstance(value, self.__class__.R):
            raise ValueError('Attribute required: '+key)
        else:
            return value
    
    def __setitem__(self, k, v):
        if not isinstance(k, str):
            raise ValueError('Key must be string')
        elif k.endswith('.'):
            self.__setitem__(k[:-1],self.__class__(v))
            return
        else:
            for ks in k.split('.'):
                if not ks.isidentifier():
                    raise ValueError('Key must be valid identifier: '+k)
        
        if isinstance(v, HDict):
            self.update({k+'.'+kv:vv for kv,vv in v.items()})
        else:
            super().__setitem__(k, v)
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)
    
    def __getnode__(self, name):
        s = name+'.'
        l = len(s)
        d = {k[l:]:v for k,v in self.items() if k.startswith(s)}
        hd = self.__class__(d)
        hd.__dict__['__P__'] = self
        return hd

    def update(self, *args, **kwargs):
        val_dict = dict(*args, **kwargs)
        for k,v in val_dict.items():
            self.__setitem__(k,v)

    def copy(self):
        return self.__class__(self)
    
    def to_dict(self, exclude=()):
        d = {}
        for key in self.keys():
            if key in exclude: continue
            keys = key.split('.')
            end = len(keys) - 1
            d_ = d
            h = self
            for i, kk in enumerate(keys):
                if i == end:
                    d_[kk] = getattr(h,kk)
                else:
                    h = getattr(h,kk)
                    kk = kk+'.'
                    if kk not in d_:
                        d_[kk] = {}
                    d_=d_[kk]
        return d
    
    def to_namespace(self, exclude=()):
        d = SimpleNamespace()
        for key in self.keys():
            if key in exclude: continue
            keys = key.split('.')
            end = len(keys) - 1
            d_ = d
            h = self
            for i, kk in enumerate(keys):
                if i == end:
                    setattr(d_, kk, getattr(h,kk))
                else:
                    h = getattr(h,kk)
                    if not hasattr(d_,kk):
                        setattr(d_, kk, SimpleNamespace())
                    d_=getattr(d_,kk)
                    kk = kk+'.'
        return d
    
    def to_flatdict(self):
        d = {}
        for key in self.keys():
            h = self
            for kk in key.split('.'):
                h = getattr(h,kk)
            d[key] = h
        return d

    
    def update_from(self, *args, **kwargs):
        val_dict = self.__class__(*args, **kwargs)
        for k in val_dict.keys():
            if k not in self.keys():
                raise KeyError('No such key: '+k)
        self.update(val_dict)
    
    def inherit_from(self, *args, **kwargs):
        val_dict = self.__class__(*args, **kwargs)
        for k in sorted(val_dict.keys(),key=lambda k: len(k)):
            k_p = k.split('.')
            v = val_dict[k]
            exists = False
            for dk in self.keys():
                dk_p = dk.split('.')
                if len(k_p) > len(dk_p): continue

                bb = 0
                for k_p_, dk_p_ in zip(k_p, dk_p):
                    if k_p_ != dk_p_: break
                    else: bb += 1
                
                ee = 0
                for k_p_, dk_p_ in zip(reversed(k_p[bb:]),
                                       reversed(dk_p[bb:])):
                    if k_p_ != dk_p_: break
                    else: ee += 1

                if bb+ee==len(k_p):
                    exists = True
                    self.__setitem__(dk, v)
                
            if not exists:
                raise KeyError('No such key: '+k)
    