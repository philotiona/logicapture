"""Backend-independent visual representations. No concept vocabulary or labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
import numpy as np


def _matrix(value, count, name):
    result=np.array(value,dtype=np.float64,copy=True)
    if result.ndim!=2 or result.shape[0]!=count or result.shape[1]<1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite [number of images, dimensions] matrix")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FeatureBatch:
    image_ids: tuple[str,...]
    global_features: np.ndarray
    patch_features: tuple[np.ndarray,...] | None = None
    encoder_id: str = "unspecified"
    metadata: dict[str,Any] = field(default_factory=dict)
    extra_views: dict[str,np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        ids=tuple(self.image_ids)
        if any(not isinstance(item,str) or not item for item in ids) or len(set(ids))!=len(ids):
            raise ValueError("Image ids must be nonempty and unique within a feature batch")
        if not isinstance(self.encoder_id,str) or not self.encoder_id:
            raise ValueError("An encoder identity is required")
        object.__setattr__(self,"image_ids",ids)
        object.__setattr__(self,"global_features",_matrix(self.global_features,len(ids),"global_features"))
        if self.patch_features is not None:
            if len(self.patch_features)!=len(ids): raise ValueError("Provide one patch array per image")
            patches=tuple(_matrix(item,len(item),"patch_features") for item in self.patch_features)
            if any(item.shape[0]<1 for item in patches) or len({item.shape[1] for item in patches})>1:
                raise ValueError("Patch arrays need at least one token and a shared feature dimension")
            object.__setattr__(self,"patch_features",patches)
        if any(not key or key in ("global","patch_mean","patch_statistics","patch_spatial") for key in self.extra_views):
            raise ValueError("Extra feature views need unique non-reserved names")
        object.__setattr__(self,"extra_views",{name:_matrix(value,len(ids),name) for name,value in self.extra_views.items()})
        object.__setattr__(self,"metadata",dict(self.metadata))

    def subset(self,indices):
        indices=np.asarray(indices,dtype=int)
        return FeatureBatch(tuple(self.image_ids[i] for i in indices),self.global_features[indices],
                            tuple(self.patch_features[i] for i in indices) if self.patch_features is not None else None,
                            self.encoder_id,self.metadata,{key:value[indices] for key,value in self.extra_views.items()})

    @classmethod
    def concatenate(cls,batches):
        """Join image rows without changing the encoder or view definitions."""
        batches=tuple(batches)
        if not batches: raise ValueError("Provide at least one feature batch")
        first=batches[0]
        if any(batch.encoder_id!=first.encoder_id or batch.global_features.shape[1]!=first.global_features.shape[1]
               or set(batch.extra_views)!=set(first.extra_views)
               or (batch.patch_features is None)!=(first.patch_features is None) for batch in batches):
            raise ValueError("Concatenation requires matching encoder and feature views")
        return cls(tuple(image_id for batch in batches for image_id in batch.image_ids),
                   np.concatenate([batch.global_features for batch in batches]),
                   tuple(array for batch in batches for array in batch.patch_features) if first.patch_features is not None else None,
                   first.encoder_id,first.metadata,
                   {key:np.concatenate([batch.extra_views[key] for batch in batches]) for key in first.extra_views})

    def views(self):
        result={"global":self.global_features,**self.extra_views}
        if self.patch_features is not None and self.image_ids:
            result["patch_mean"]=np.stack([array.mean(axis=0) for array in self.patch_features])
            result["patch_statistics"]=np.stack([np.concatenate((array.mean(axis=0),array.std(axis=0))) for array in self.patch_features])
            sizes=[int(round(len(array)**.5)) for array in self.patch_features]
            if all(side>=2 and side*side==len(array) for side,array in zip(sizes,self.patch_features)):
                pooled=[]
                for side,array in zip(sizes,self.patch_features):
                    grid=array.reshape(side,side,-1)
                    boundary=side//2
                    pooled.append(np.concatenate([grid[a:b,c:d].mean(axis=(0,1))
                        for a,b in ((0,boundary),(boundary,side)) for c,d in ((0,boundary),(boundary,side))]))
                result["patch_spatial"]=np.stack(pooled)
        return result


class VisualEncoder(Protocol):
    encoder_id: str
    def encode(self,images:Sequence[Any],image_ids:Sequence[str]|None=None)->FeatureBatch: ...


def fuse_feature_batches(batches,encoder_id=None):
    """Fuse named representations of the SAME images, retaining individual views.

    Alignment is by explicit image id. Each global block is unit normalized so
    a high-dimensional or differently-scaled encoder cannot dominate by units.
    View selection and any standardization still occur within support folds.
    """
    if not batches: raise ValueError("Provide at least one named representation")
    names=sorted(batches)
    first=batches[names[0]]
    blocks=[];views={};components={};automatic=["global"]
    for name in names:
        batch=batches[name]
        if set(batch.image_ids)!=set(first.image_ids): raise ValueError("Fusion image ids must match")
        positions={value:index for index,value in enumerate(batch.image_ids)}
        batch=batch.subset([positions[value] for value in first.image_ids])
        array=batch.global_features
        blocks.append(array/np.maximum(np.linalg.norm(array,axis=1,keepdims=True),1e-12))
        views.update({f"{name}:{view}":array for view,array in batch.views().items()})
        allowed=batch.metadata.get("automatic_feature_views",tuple(batch.views()))
        if any(view not in batch.views() for view in allowed): raise ValueError("Unknown automatic feature view")
        automatic.extend(f"{name}:{view}" for view in allowed)
        components[name]={"encoder_id":batch.encoder_id,"metadata":batch.metadata}
    identity=encoder_id or "fusion["+",".join(f"{name}={batches[name].encoder_id}" for name in names)+"]"
    return FeatureBatch(first.image_ids,np.concatenate(blocks,axis=1)/len(blocks)**.5,
                        encoder_id=identity,metadata={"components":components,"fusion":"equal unit-norm global blocks",
                                                      "automatic_feature_views":automatic},extra_views=views)


@dataclass(frozen=True)
class EpisodeSupport:
    features: FeatureBatch
    labels: tuple[bool,...]
    episode_id: str = ""

    def __post_init__(self):
        labels=tuple(self.labels)
        if len(labels)!=len(self.features.image_ids) or any(not isinstance(value,(bool,np.bool_)) for value in labels):
            raise ValueError("Support labels must be booleans aligned with image ids")
        if set(labels)!={False,True}: raise ValueError("An episode requires both support groups")
        object.__setattr__(self,"labels",tuple(bool(value) for value in labels))
