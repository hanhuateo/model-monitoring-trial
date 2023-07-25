from dataclasses import dataclass

@dataclass(frozen=True)
class DatkStatsResult:
    statistic: float
    pvalue: float
    significant: bool

@dataclass(frozen=True)
class DatkNormResult:
    pvalue: float
    normal: bool

@dataclass(frozen=True)
class DatkEqualVarResult:
    pvalue: float
    equal_variance: bool

@dataclass(frozen=True)
class DatkTTestResult(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkMannWhitneyUResult(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkChi2Result(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkFishersTestResult(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkOneWayAnovaResult(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkKruskalWallisResult(DatkStatsResult):
    pass        

@dataclass(frozen=True)
class DatkPearsonrResult(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkSpearmanrResult(DatkStatsResult):
    pass

@dataclass(frozen=True)
class DatkPostHocResult:
    combi: tuple
    pvalue: float
    significant: bool

@dataclass(frozen=True)
class DatkChi2PostHocResult(DatkPostHocResult):
   corrected_pvalue: float