package statistics

import (
    "log"
    "math"
)

type FeaturesAnalyzer struct {
}

func NewFeaturesAnalyzer() FeaturesAnalyzer {
    return FeaturesAnalyzer{}
}

func (f *FeaturesAnalyzer) Analyze(samples [][]float64) (statistics []FeatureStatistic, distribution ClassDistribution) {
    numberOfSamples := len(samples)
    if numberOfSamples < 1 {
        log.Fatal("At least one sample is needed to analyze.")
    }
    var numberOfFeatures = len(samples[0])
    if numberOfFeatures < 1 {
        log.Fatal("At least feature is needed to analyze.")
    }
    for i := 0; i < numberOfFeatures; i++ {
        statistics = append(statistics, NewFeatureStatistic())
    }
    for _, sample := range samples {

        // Find class distribution.
        y := sample[len(sample) - 1]
        if y == -1 {
            distribution.Negative++
        } else {
            distribution.Positive++
        }

        // Find min and max
        for i := 0; i < numberOfFeatures; i++ {
            statistic := &statistics[i]
            featureValue := sample[i]
            if featureValue < statistic.Min {
                statistic.Min = featureValue
            }
            if featureValue > statistic.Max {
                statistic.Max = featureValue
            }
            statistic.Sum += featureValue
        }
    }

    // Find avg abd rng
    for i, _ := range statistics {
        statistic := &statistics[i]
        statistic.Avg = statistic.Sum / float64(numberOfSamples)
        statistic.Rng = math.Abs(statistic.Max - statistic.Min)
    }

    // Find variance
    for _, sample := range samples {
        for i := 0; i < numberOfFeatures; i++ {
            statistic := &statistics[i]
            featureValue := sample[i]
            statistic.Vrn += math.Pow(statistic.Avg - featureValue, 2)
        }
    }

    // Find std
    for i, _ := range statistics {
        statistic := &statistics[i]
        statistic.Vrn /= float64(numberOfSamples - 1)
        statistic.Std = math.Sqrt(statistic.Vrn)
    }
    return
}

/**
 * Computes covariance and correlation.
 */
func (f *FeaturesAnalyzer) Correlation(x, y uint, samples [][]float64, statistics []FeatureStatistic) VariableRelations {
    sum := 0.0
    for _, sample := range samples {
        xValue := float64(sample[x])
        yValue := float64(sample[y])
        sum += (xValue - statistics[x].Avg) * (yValue - statistics[y].Avg)
    }
    cov := sum / float64(len(samples) - 1)
    cor := cov / (statistics[x].Std * statistics[y].Std)
    return VariableRelations{X: x, Y: y, Cov: cov, Cor: cor}
}
