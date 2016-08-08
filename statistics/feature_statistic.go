package statistics

import "math"

type FeatureStatistic struct {
    Min float64
    Max float64
    Sum float64
    Avg float64
    Vrn float64
    Std float64
    Rng float64
}

func NewFeatureStatistic() FeatureStatistic {
    return FeatureStatistic{Min: math.MaxFloat64, Max: -math.MaxFloat64}
}
