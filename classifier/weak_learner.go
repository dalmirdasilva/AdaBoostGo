package classifier

import (
    "github.com/dalmirdasilva/AdaBoostGo/statistics"
    "github.com/dalmirdasilva/AdaBoostGo/config"
    "math/rand"
    "math"
    "log"
    "fmt"
)

type WeakLearner struct {
    analyzer         statistics.FeaturesAnalyzer
    classifiersCache []WeakClassifier
}

func NewWeakLearner() WeakLearner {
    return WeakLearner{analyzer: statistics.NewFeaturesAnalyzer(), classifiersCache: []WeakClassifier{}}
}

// Uses FeaturesAnalyzer to analyze the samples.
// It computes min, max, avg, std...
func (w *WeakLearner) analyzeFeatures(samples [][]float64) []statistics.FeatureStatistic {
    statistics, _ := w.analyzer.Analyze(samples)
    return statistics
}

// Learn weak classifier h_{t} using distribution D_{t}.
//
// Compute the weighted error for each weak classifier.
// Select the weak classifier with minimum error.
//
// Implementation of the following equation:
// h_{t} = \underset {h_{j}\in H}{\operatorname {arg\,min} }\,\epsilon_{j} = \sum_{i=1}^{m}D_{t}\left [y_{i} \neq h_{j}(x_{i}) \right ]
//
func (w *WeakLearner) GenerateWeakClassifier(samples [][]float64, weights []float64) WeakClassifier {

    numberOfSamples := len(samples)
    if numberOfSamples < 1 {
        log.Fatal("At least one sample is needed to generate.")
    }

    numberOfFeatures := uint(len(samples[0]) - 1)
    if numberOfFeatures < 1 {
        log.Fatal("At least feature is needed to generate.")
    }

    var classifiers *[]WeakClassifier

    // Generate the weak classifiers.
    if config.USE_RANDOM_WEAK_CLASSIFIERS {
        classifiers = w.generateRandomClassifiers(samples, numberOfFeatures)
    } else {
        classifiers = w.generateAllPossibleClassifiers(samples, numberOfFeatures)
    }

    var bestIndex int
    bestError := math.MaxFloat64

    // Error minimization step.
    // For each random classifier:
    for i, _ := range *classifiers {
        classifier := &(*classifiers)[i]
        classifier.SetError(0.0)
        for j, sample := range samples {
            y := sample[len(sample) - 1]

            // If wrongly classified.
            if float64(classifier.Classify(sample)) != y {

                // Sums the sample's weight to its error.
                classifier.IncreaseError(weights[j])
            }
        }

        // Retains the classifier with minor error.
        if classifier.GetError() < bestError {
            bestError = classifier.GetError()
            bestIndex = i
        }
    }

    best := (*classifiers)[bestIndex]

    // Remove the best classifier from the list.
    // It is needed only when using non-random classifiers.
    // When using random, the list is recreated each step. Otherwise the
    // list is reused over and over again. Removing used classifier is needed
    // to avoid piking the same classifiers more than once.
    if !config.USE_RANDOM_WEAK_CLASSIFIERS {

        // Replace the element at the best index to the last one and then remove the last.
        (*classifiers)[bestIndex] = (*classifiers)[len(*classifiers) - 1]
        *classifiers = (*classifiers)[:len(*classifiers) - 1]
    }
    return best
}

// Generates a bunch of random weak classifiers.
func (w *WeakLearner) generateRandomClassifiers(samples [][]float64, numberOfFeatures uint) *[]WeakClassifier {

    var classifiers []WeakClassifier

    // Analyses the given samples. Computes min, max, avg, std...
    featuresMetrics := w.analyzeFeatures(samples)

    // Creates Config.NUMBER_OF_RANDOM_CLASSIFIERS random classifiers.
    for i := 0; i < config.NUMBER_OF_RANDOM_CLASSIFIERS; i++ {

        // Random feature number.
        featureNumber := uint(rand.Intn(int(numberOfFeatures)))

        // Gets info about the feature.
        info := featuresMetrics[featureNumber]

        // Use the info to randomly choose the split value.
        split := (rand.Float64() * info.Rng) + info.Min

        // Creates and append the random classifier into the list.
        classifiers = append(classifiers, NewWeakClassifier(featureNumber, split))
    }
    return &classifiers
}

// Generates all possibilities of classifiers. For all unique positions in the trainingSet it
// generates a weak classifiers.
func (w *WeakLearner) generateAllPossibleClassifiers(samples [][]float64, numberOfFeatures uint) *[]WeakClassifier {

    // All possible classifiers can be computed one and then stored.
    if len(w.classifiersCache) == 0 {

        // This matrix stores all features, for each feature (first dimension) it stores all
        // different values the feature has in the training set.
        var matrix []map[float64]bool
        for i := uint(0); i < numberOfFeatures; i++ {
            matrix = append(matrix, make(map[float64]bool))
        }
        for _, sample := range samples {
            for j := uint(0); j < numberOfFeatures; j++ {
                sampleValue := sample[j]
                matrix[j][sampleValue] = true
            }
        }
        for featureIndex, entry := range matrix {
            fmt.Println(len(entry))
            for featureValue, _ := range entry {
                w.classifiersCache = append(w.classifiersCache, NewWeakClassifier(uint(featureIndex), featureValue))
            }
        }
    }
    return &w.classifiersCache
}