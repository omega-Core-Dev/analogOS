"""
analogOS · pipeline.py
Pipeline universal — encadeia scan → broadcast → candidate → propagate → compose.
Qualquer domínio usa a mesma interface; só os parâmetros mudam.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from analog_core.entity import Entity
from analog_core.primitives.scan import scan
from analog_core.primitives.broadcast import broadcast
from analog_core.primitives.candidate import candidate
from analog_core.primitives.propagate import propagate
from analog_core.primitives.compose import compose


@dataclass
class PipelineConfig:
    # broadcast
    intensity: float = 1.0
    falloff: str = "sqrt"
    noise: float = 0.0
    # candidate
    threshold: float = 0.4
    # propagate
    social_factor: float = 0.5
    radius: float | None = None
    # compose
    mode: str = "top_k"
    top_k: int = 3


@dataclass
class PipelineResult:
    ref_map: dict
    signals_broadcast: dict[str, float]
    candidates: list[Entity]
    signals_propagated: dict[str, float]
    output: dict
    # cluster usado no compose
    cluster: list[Entity] = field(default_factory=list)


class Pipeline:
    """
    Interface universal do analogOS.

    Uso:
        pipeline = Pipeline(config=PipelineConfig(threshold=0.3, mode='top_k'))
        result = pipeline.run(source=query_entity, entities=memory_entities)
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        scan_key_fn: Callable[[Entity], str] | None = None,
    ):
        self.config = config or PipelineConfig()
        self.scan_key_fn = scan_key_fn

    def run(self, source: Entity, entities: list[Entity]) -> PipelineResult:
        cfg = self.config

        # 1. SCAN — constrói mapa de referência
        ref_map = scan(entities, key_fn=self.scan_key_fn)

        # 2. BROADCAST — fonte emite sinal
        signals_bc = broadcast(
            source=source,
            ref_map=ref_map,
            intensity=cfg.intensity,
            falloff=cfg.falloff,
            noise=cfg.noise,
        )

        # 3. CANDIDATE — filtra elegíveis
        candidates = candidate(
            entities=entities,
            signals=signals_bc,
            threshold=cfg.threshold,
        )

        # 4. PROPAGATE — candidatos repassam para vizinhos
        signals_prop = propagate(
            candidates=candidates,
            all_entities=entities,
            signals=signals_bc,
            social_factor=cfg.social_factor,
            radius=cfg.radius,
        )

        # cluster = candidatos + entidades que receberam propagação acima de threshold/2
        half_thresh = cfg.threshold / 2
        candidate_ids = {e.id for e in candidates}
        propagated_extras = [
            e for e in entities
            if e.id not in candidate_ids
            and signals_prop.get(e.id, 0.0) >= half_thresh
        ]
        cluster = candidates + propagated_extras

        # 5. COMPOSE — agrega cluster em saída
        output = compose(
            cluster=cluster,
            signals=signals_prop,
            mode=cfg.mode,
            top_k=cfg.top_k,
        )

        return PipelineResult(
            ref_map=ref_map,
            signals_broadcast=signals_bc,
            candidates=candidates,
            signals_propagated=signals_prop,
            cluster=cluster,
            output=output,
        )
