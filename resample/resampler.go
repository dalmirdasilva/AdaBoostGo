package resample

import (
    "github.com/dalmirdasilva/GoAdaboostClassifier/statistics"
    "math"
)

type Resampler struct {
    distribution *statistics.ClassDistribution
}

func NewResampler() Resampler {
    return Resampler{}
}

func (r *Resampler) OverSample(samples [][]float64) [][]float64 {
    distribution := r.getClassDistribution(samples)
    y0 := distribution.Negative
    y1 := distribution.Positive
    majority := -1.0
    if y0 < y1 {
        majority = 1.0
    }
    difference := math.Abs(float64(y0) - float64(y1))
    for _, sample := range samples {
        if difference <= 0 {
            break
        }
        if sample[len(sample) - 1] != majority {
            samples = append(samples, sample)
            difference--
        }
    }
    return samples
}

func (r *Resampler) getClassDistribution(instances [][]float64) statistics.ClassDistribution {
    analyzer := statistics.NewFeaturesAnalyzer()
    _, distribution := analyzer.Analyze(instances)
    return distribution
}