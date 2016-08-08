package classifier

import (
    "github.com/dalmirdasilva/AdaBoostGo/statistics"
    "github.com/dalmirdasilva/AdaBoostGo/resample"
    "github.com/dalmirdasilva/AdaBoostGo/config"
    "math"
    "fmt"
)

type AdaBoost struct {
    WeakClassifiers     []WeakClassifier
    weakLearner         WeakLearner
    numberOfClassifiers uint
    weights             []float64
}

func NewAdaBoost(numberOfClassifiers uint) AdaBoost {
    return AdaBoost{
        weakLearner: NewWeakLearner(),
        WeakClassifiers: []WeakClassifier{},
        numberOfClassifiers: numberOfClassifiers,
        weights: []float64{},
    }
}

// All weights should be initialized with the same distribution.
func (c *AdaBoost) initializeWeights(samples [][]float64) {
    samplesLength := uint(len(samples))
    c.weights = make([]float64, samplesLength)
    var positiveWeight, negativeWeight float64
    negativeWeight = 1 / float64(samplesLength)
    positiveWeight = negativeWeight
    if config.INCORPORATE_COST_SENSITIVE_LEARNING {
        analyzer := statistics.NewFeaturesAnalyzer()
        _, distribution := analyzer.Analyze(samples)
        positiveRate := float64(distribution.Positive) / float64(samplesLength)
        negativeRate := float64(distribution.Negative) / float64(samplesLength)
        normalizingConstant := (float64(distribution.Negative) * positiveRate) + (float64(distribution.Positive) * negativeRate)

        // Weights are inversely proportional to the rate.
        // The class with less occurrences gets higher weight.
        positiveWeight = positiveRate / float64(normalizingConstant)
        negativeWeight = negativeRate / float64(normalizingConstant)
    }
    for i, sample := range samples {
        y := sample[len(sample) - 1];
        if y == -1 {
            c.weights[i] = positiveWeight
        } else {
            c.weights[i] = negativeWeight
        }
    }
}

// Update the distribution based on the performance.
//
// Computes the following equation:
// D_{t+1}(i)=\frac{D_{t}(i)e(-\alpha_{t}y_{i}h_{t}(x_{i}))}{Z_{t}}
//
// where Z t is a normalization factor to keep D_{t+1} a distribution. Note the careful evaluation of the term inside of
// the exp based on the possible {−1, +1} values of the label.
func (c *AdaBoost) updateWeights(classifier WeakClassifier, samples[][]float64) {
    sum := float64(0)
    for i, sample := range samples {
        y := sample[len(sample) - 1]

        // D_{t+1}(i)=\frac{D_{t}(i)e(-\alpha_{t}y_{i}h_{t}(x_{i}))}{Z_{t}}.
        c.weights[i] *= math.Exp(-(classifier.GetAlpha()) * float64(classifier.Classify(sample)) * y)

        // Summing up the Z_{t}.
        sum += c.weights[i]
    }

    // Dividing each D to Z_{t}.
    for i := 0; i < len(c.weights); i++ {
        c.weights[i] /= sum
    }
}


// Learn!
//
// @param samples
func (c *AdaBoost) Train(samples [][]float64) {

    if config.OVER_SAMPLING_TRAINING_SET {
        resampler := resample.NewResampler()
        samples = resampler.OverSample(samples)
    }

    c.initializeWeights(samples)

    // Build T classifiers.
    for i := uint(0); i < c.numberOfClassifiers; i++ {

        fmt.Println("boom")

        // Call the learner and receive the built classifier.
        weakClassifier := c.weakLearner.GenerateWeakClassifier(samples, c.weights)

        // Computes the alpha for the built classifier.
        weakClassifier.ComputeAlpha()

        // Updates the weights.
        c.updateWeights(weakClassifier, samples)

        // Save the classifier.
        c.WeakClassifiers = append(c.WeakClassifiers, weakClassifier)

    }
}

// H(x)=sign(\sum_{t=1}^{T}{\alpha_{t}h_{t}(x)})
func (c *AdaBoost) Classify(sample []float64) (score float64) {
    for _, weakClassifier := range c.WeakClassifiers {
        score += weakClassifier.ClassifyWithAlpha(sample)
    }
    return
}